/*
 * Windows-faithful `findstr` for the F4.5 harness.
 *
 * Wine's findstr treats a bare "^" as a LITERAL caret; real Windows findstr
 * defaults to regex, so `findstr "^"` matches every line. start.cmd relies on
 * the Windows behaviour twice ("is this folder empty?", "is the container
 * running?"), and under Wine's version both always answered "no lines" — which
 * would have been read as a defect in start.cmd. It is not; it is the
 * reimplementation. This shim restores Windows semantics for exactly the three
 * invocations start.cmd makes, so the harness tests the launcher and not Wine.
 *
 * Supported:  findstr "^"                       -> any line matches
 *             findstr /b /c:"LITERAL" <file>    -> lines starting with LITERAL
 * Anything else is reported loudly rather than guessed at.
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    const char *lit = NULL, *pat = NULL, *file = NULL;
    int at_begin = 0, i, found = 0;
    char line[8192];

    for (i = 1; i < argc; i++) {
        if (strncmp(argv[i], "/c:", 3) == 0)      lit = argv[i] + 3;
        else if (_stricmp(argv[i], "/b") == 0)    at_begin = 1;
        else if (_stricmp(argv[i], "/r") == 0)    ;
        else if (argv[i][0] == '/')               ;
        else if (!lit && !pat)                    pat = argv[i];
        else                                      file = argv[i];
    }
    if (!lit && pat && strcmp(pat, "^") != 0) {
        fprintf(stderr, "[harness findstr] unsupported pattern '%s'\n", pat);
        return 2;
    }

    FILE *in = stdin;
    if (file && !(in = fopen(file, "r"))) return 2;

    while (fgets(line, sizeof line, in)) {
        char *nl = strpbrk(line, "\r\n");
        if (nl) *nl = 0;
        int hit;
        if (lit) hit = at_begin ? (strncmp(line, lit, strlen(lit)) == 0)
                                : (strstr(line, lit) != NULL);
        else     hit = 1;                       /* the "^" case: every line */
        if (hit) { found = 1; printf("%s\n", line); }
    }
    if (in != stdin) fclose(in);
    return found ? 0 : 1;
}
