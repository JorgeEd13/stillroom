@echo off
REM Start the assistant. Double-click this file.
REM
REM The Windows twin of start.sh. Kept as a separate file rather than one
REM cross-platform script because a client double-clicks this one, and a .cmd
REM that shells out to something else is a thing their antivirus asks about.
REM
REM `pause` at every exit is deliberate: double-clicked windows close instantly,
REM taking the explanation with them.

setlocal
cd /d "%~dp0"

echo ------------------------------------------------------------
echo   Starting your document assistant
echo ------------------------------------------------------------
echo.

REM --- 1. Docker ----------------------------------------------------------
REM ⚠️ Two questions, not one (F4.5). `docker info` fails identically whether
REM Docker is missing or merely stopped, and the single check that used to be
REM here opened with "Docker is not running ... open Docker Desktop and wait
REM until it says it is running" — addressed to somebody who may not have it
REM installed at all. start.sh has always asked both questions; this file asked
REM one and guessed, which is the same wrong-computer reasoning in a smaller key.
where docker >nul 2>&1
if errorlevel 1 (
  echo Docker is not installed on this computer.
  echo.
  echo The assistant runs inside Docker. Install Docker Desktop from Docker's
  echo own website, start it, then double-click this file again.
  echo.
  pause
  exit /b 1
)

docker info >nul 2>&1
if errorlevel 1 (
  echo Docker is installed, but it is not running.
  echo.
  echo Open Docker Desktop, wait until it says it is running, then
  echo double-click this file again.
  echo.
  pause
  exit /b 1
)

REM --- 1b. Compose, BEFORE anything that uses Compose ----------------------
REM ⛔ F4.5, and it made the check dead code in the only situation it exists
REM for. This used to sit *after* the two model probes — and both probes are
REM `docker compose run`. On a machine with Docker but no Compose plugin the
REM probe failed first, so the client was told their Ollama networking was
REM broken and asked to send a screenshot of "something unusual on Windows",
REM when the true answer was "install the Compose plugin". A check that can
REM only be reached when it would pass is not a check.
docker compose version >nul 2>&1
if errorlevel 1 (
  echo This computer has Docker, but not Docker Compose.
  echo.
  echo Compose is included with Docker Desktop. If Docker was installed some
  echo other way, install the Compose plugin, then try again.
  echo.
  pause
  exit /b 1
)

REM --- 2. Ollama ----------------------------------------------------------
REM ⚠️ Probed from inside a container, not from here (leg B #12). Checking
REM localhost from Windows answers a question about the WRONG COMPUTER: the
REM assistant runs in a container, and that is what has to reach the model.
REM Docker Desktop normally proxies host loopback so this passes where the
REM Linux equivalent does not — but "normally" is not "verified", and the whole
REM campaign has been about the gap between those two words.
REM
REM ⛔ And the probe used to be on the WRONG NETWORK too: a bare
REM `docker run` lands on the DEFAULT bridge, not the network Compose builds
REM for this project. Measured on Linux: default bridge HTTP 200, compose
REM network HTTP 000, preflight green, product dead. `docker compose run`
REM puts the probe on the project network with the same address and the same
REM extra_hosts as the assistant, because they share one anchor in the
REM compose file.
echo Checking the AI model is reachable...
docker compose run --rm probe >nul 2>&1
if errorlevel 1 (
  REM Two different worlds, opposite actions. Telling a client to start
  REM software that is already running is how a support thread starts.
  curl -s -f -m 5 http://localhost:11434/api/tags >nul 2>&1
  if errorlevel 1 (
    echo Ollama is not responding on this computer.
    echo.
    echo Ollama is what runs the AI model locally - it is the reason your
    echo documents never leave this machine. Start Ollama, wait a few seconds,
    echo then double-click this file again.
  ) else (
    echo Ollama is running, but the assistant cannot reach it.
    echo.
    echo Ollama is only listening to this computer itself, and the assistant
    echo runs in its own container, which counts as somewhere else.
    echo.
    echo Please send me this screen - this is unusual on Windows and I will
    echo tell you the exact setting to change.
  )
  echo.
  pause
  exit /b 1
)

REM --- 2a. The SECOND model, which the client has no reason to know about --
REM ⚠️ New with the move to bge-m3, and the likeliest first-run failure of any delivery
REM after it. Reading the documents used to need nothing but the image; it now
REM needs bge-m3 on their Ollama. A client who did exactly what the runbook told
REM them still fails - inside the build, on a name they have never seen.
REM
REM Same compose-service probe as above: same network, same address,
REM same anchor. It asks for an EMBEDDING, not the tag list - a tag proves the
REM weights are on disk, only an embedding proves the server can load them.
REM
REM ⚠️ NOT VERIFIED ON WINDOWS. This file has still never been executed by
REM anybody; F4.5 exercises its batch logic under Wine, which proves "no gross
REM batch error" and NOT "runs on Windows".
echo Checking the model that reads your documents...
docker compose run --rm probe-embedding >nul 2>&1
if errorlevel 1 (
  echo The assistant needs a second AI model, and it is not installed yet.
  echo.
  echo There are two: one writes the answers, and one reads your documents so
  echo the right passage can be found. Only the first one is installed.
  echo.
  echo Install the second one - this downloads about 1.2 GB, once:
  echo.
  echo        ollama pull bge-m3:567m
  echo.
  echo Then double-click this file again.
  echo.
  pause
  exit /b 1
)

REM --- 2b. The things that make the first build fail -----------------------
REM (The Compose check that used to be here moved to 1b — see the note there.)
if not exist "documents\" (
  echo The 'documents' folder is not next to this file.
  echo.
  echo The assistant answers questions about the documents in that folder.
  echo.
  pause
  exit /b 1
)
dir /b /a "documents" 2>nul | findstr "^" >nul
if errorlevel 1 (
  echo The 'documents' folder is empty.
  echo.
  echo Put the documents you want the assistant to answer about in there,
  echo then double-click this file again.
  echo.
  pause
  exit /b 1
)

REM --- 2b2. The prepared image, from the file next to this one -------------
REM ⚠️ Must come before anything that builds - including the document check
REM below, which runs `docker compose run` and will build the image first.
REM Loading it here is what makes the first build touch nothing but the
REM client's own Ollama. Without it the build pulls ~90 libraries
REM from PyPI and an embedding model from another company's cloud, which was
REM OBSERVED failing.
REM BASE_TAG, not BASE_IMAGE: Compose reads BASE_IMAGE from the environment for
REM the build argument of the same name, and setting it here would hand the
REM build a tag it must resolve from a registry it cannot reach.
set BASE_TAG=
for /f "tokens=2 delims==" %%v in ('findstr /b /c:"ARG BASE_IMAGE=" Dockerfile') do set BASE_TAG=%%v
if not defined BASE_TAG set BASE_TAG=stillroom-base:0.1.0

REM ⛔ This section used to be a `call :load_base` subroutine, and that was a
REM defect with teeth (F4.5). `exit /b 1` inside a CALLed label returns from the
REM SUBROUTINE, not from the script — so the corrupt-tarball path printed "the
REM prepared file could not be read ... ask me for it again", paused, and then
REM carried straight on. The client pressed a key and watched it either declare
REM "Ready. Opening http://localhost:8000" and exit 0, or blame their document
REM count for a build failure caused by the missing base image. start.sh has no
REM such bug because it uses an inline if/else, where `exit 1` ends the script.
REM The fix is to delete the construct rather than work around it: there is no
REM `call` here now, so every `exit /b 1` is a real exit.
docker image inspect %BASE_TAG% >nul 2>&1
if not errorlevel 1 goto :after_base

REM ⚠️ The FIRST match, not the last. `dir /b` sorts by name, and the bare
REM `set TARBALL=%%f` that used to be here kept the LAST one — so a client who
REM had been sent 0.1.0 and later dropped 0.2.0 into the same folder loaded
REM 0.2.0 while the build asked for the 0.1.0 that `ARG BASE_IMAGE` pins, and
REM the load was wasted. start.sh takes `| head -1`; the two launchers must not
REM disagree about which file they picked, or no support thread is reproducible.
set TARBALL=
for /f "delims=" %%f in ('dir /b stillroom-base-*.tar.gz 2^>nul') do if not defined TARBALL set TARBALL=%%f

if defined TARBALL goto :load_base

REM Says so and continues, rather than refusing - same judgement as the model
REM probe above. A client with the internet is fine; a client without one now
REM knows what to ask for instead of reading a Python traceback.
echo The prepared assistant file is not in this folder, so this first run
echo will download what it needs from the internet. It usually works. If it
echo stops with an error, send me this screen - I will send you the file
echo and it will not need the internet at all.
echo.
REM Names the Dockerfile's own `base` STAGE, so the build makes it here.
set BASE_IMAGE=base
goto :after_base

:load_base
echo Preparing the assistant - this happens once, and takes a minute...
docker load -i "%TARBALL%" >nul 2>&1
if errorlevel 1 (
  echo.
  echo The prepared file '%TARBALL%' could not be read. It is usually
  echo incomplete, which happens when a download or a copy is interrupted.
  echo.
  echo Delete it, ask me for it again, and put the new one next to this file.
  echo.
  pause
  exit /b 1
)

:after_base

REM --- 2c. Documents changed? Re-read them BEFORE starting -----------------
REM ⚠️ Re-running this file used to do nothing for a client who had added a
REM document: `docker compose up -d` does not rebuild an existing image, so the
REM new file was never indexed and nothing said so.
REM Stopping first is required, not tidiness — the running service pins its
REM corpus at startup, and re-indexing underneath it made every question fail.
docker compose ps --status running --quiet assistant 2>nul | findstr "^" >nul
if not errorlevel 1 (
  echo Stopping the assistant so the documents can be re-read...
  docker compose stop assistant >nul 2>&1
)

echo Checking your documents...
docker compose run --rm --no-deps assistant stillroom refresh --config client.toml
if errorlevel 1 (
  echo.
  echo Your documents could not be read. The assistant has NOT been changed -
  echo it will keep answering from the documents it already knew.
  echo.
  echo The most likely cause is more documents than this assistant is set up
  echo for. The lines above say which. Send me this screen if not.
  echo.
  pause
  exit /b 1
)

REM --- 3. Start -----------------------------------------------------------
echo Starting... the first run can take a minute.
echo.
docker compose up -d
if errorlevel 1 (
  echo.
  echo The assistant could not start. Please send me everything above this
  echo line and I will tell you what it means.
  echo.
  pause
  exit /b 1
)

REM --- 4. Wait, then open the browser -------------------------------------
echo Waiting for the assistant to be ready...
setlocal enabledelayedexpansion
set READY=
set STOPPED=
for /l %%i in (1,1,60) do (
  if not defined READY if not defined STOPPED (
    curl -s -f -m 3 http://localhost:8000/api/health >nul 2>&1
    if not errorlevel 1 set READY=yes
    REM ⚠️ A crashed container and a slow one are not the same thing, and this
    REM loop used to report both as "slow" (leg B #9) — then exit 0, pointing
    REM the client at a URL that would never load.
    if not defined READY (
      set RUNNING=
      for /f %%c in ('docker compose ps --status running --quiet assistant 2^>nul') do set RUNNING=yes
      if not defined RUNNING (set STOPPED=yes) else (timeout /t 2 /nobreak >nul)
    )
  )
)

if defined STOPPED (
  echo.
  echo The assistant started and then stopped.
  echo.
  echo This is usually the AI model: the assistant checks it can reach it as
  echo it starts, and gives up if it cannot. What it said:
  echo.
  docker compose logs --tail=15 assistant
  echo.
  echo Send me everything above this line and I will tell you what it means.
  echo.
  pause
  exit /b 1
)

if not defined READY (
  echo.
  echo It is taking longer than expected. It may still be starting up.
  echo Open this address in your browser in a minute:  http://localhost:8000
  echo.
  pause
  exit /b 0
)

echo.
echo ------------------------------------------------------------
echo   Ready.  Opening http://localhost:8000
echo ------------------------------------------------------------
echo.
echo To stop it later, double-click stop.cmd.
echo.
start "" http://localhost:8000
pause
