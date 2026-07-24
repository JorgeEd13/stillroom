/*
 * F4.5 test double for `docker`, `curl` and `timeout`.
 *
 * One binary, installed under three names; it decides what it is from argv[0].
 * Behaviour comes from a plain-text rule file (%STUB_SCRIPT%) so a scenario is
 * data, not a rebuild. Every invocation is appended to %STUB_LOG%, which is
 * what proves which branch of start.cmd actually ran.
 *
 * Rule syntax, one per line:   <pattern>|<exit code>|<stdout>
 *   pattern   matched against "<tool> <args...>" joined with single spaces.
 *             A trailing '*' makes it a prefix match; otherwise it is exact.
 *             First match wins, so put the specific rule above the general one.
 *   stdout    '\n' is expanded; empty means print nothing.
 * An invocation matching no rule exits 0 and is logged [NO RULE] — that is a
 * hole in the scenario, never something to read past.
 */
#include <windows.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static void tool_name(const char *p, char *out, size_t n)
{
    const char *b = p, *q;
    for (q = p; *q; q++)
        if (*q == '\\' || *q == '/')
            b = q + 1;
    snprintf(out, n, "%s", b);
    size_t l = strlen(out);
    if (l > 4 && _stricmp(out + l - 4, ".exe") == 0)
        out[l - 4] = 0;
}

static void emit(const char *s)
{
    for (; *s; s++) {
        if (s[0] == '\\' && s[1] == 'n') { putchar('\n'); s++; }
        else putchar(*s);
    }
    putchar('\n');
}

int main(int argc, char **argv)
{
    char cmdline[8192], tool[128];
    int i, code = 0, matched = 0;
    char out[4096];

    setvbuf(stdout, NULL, _IONBF, 0);
    out[0] = 0;

    tool_name(argv[0], tool, sizeof tool);
    snprintf(cmdline, sizeof cmdline, "%s", tool);
    for (i = 1; i < argc; i++) {
        strncat(cmdline, " ", sizeof cmdline - strlen(cmdline) - 1);
        strncat(cmdline, argv[i], sizeof cmdline - strlen(cmdline) - 1);
    }

    const char *script = getenv("STUB_SCRIPT");
    if (script) {
        FILE *f = fopen(script, "r");
        if (f) {
            char buf[8192];
            while (fgets(buf, sizeof buf, f)) {
                char *nl = strpbrk(buf, "\r\n");
                if (nl) *nl = 0;
                if (!buf[0] || buf[0] == '#') continue;
                char *p1 = strchr(buf, '|');
                if (!p1) continue;
                *p1++ = 0;
                char *p2 = strchr(p1, '|');
                if (!p2) continue;
                *p2++ = 0;

                size_t pl = strlen(buf);
                int hit;
                if (pl && buf[pl - 1] == '*')
                    hit = (strncmp(cmdline, buf, pl - 1) == 0);
                else
                    hit = (strcmp(cmdline, buf) == 0);

                if (hit) {
                    code = atoi(p1);
                    snprintf(out, sizeof out, "%s", p2);
                    matched = 1;
                    break;
                }
            }
            fclose(f);
        }
    }

    if (out[0]) emit(out);

    const char *log = getenv("STUB_LOG");
    if (log) {
        FILE *g = fopen(log, "a");
        if (g) {
            const char *bi = getenv("BASE_IMAGE");
            fprintf(g, "%-62s -> exit %d%s%s%s\n", cmdline, code,
                    matched ? "" : "   [NO RULE]",
                    bi ? "   BASE_IMAGE=" : "", bi ? bi : "");
            fclose(g);
        }
    }
    return code;
}
