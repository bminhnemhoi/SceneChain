# DRES v2 notes for VBS 2027

These notes go with `scenechain/dres.py` (the client) and `tests/test_dres.py` (the mock-server
tests). They record where the protocol details come from, how a VBS session with DRES works from a
participant's point of view, what we checked against a real DRES, and how to rehearse before the event.

Status on 17 Sep 2026: the client passes its mock-server tests, and it ran end to end against a locally
started DRES 2.0.4 release (details in [section 5](#5-what-was-verified-against-a-real-dres)).

## 1. Sources

| What | Where | Version read |
|---|---|---|
| Client OpenAPI specification (the file DRES publishes for generating participant clients) | https://github.com/dres-dev/DRES/blob/v2.0.5/doc/oas-client.json | tag `v2.0.5` = commit `bd600c8d3e586e92356e9362d68634eb365e8a85` (28 May 2026), `info.version` 2.0.5-SNAPSHOT. The file last changed in commit `9059aeb2` (16 Jul 2024); `master` and `dev` hold the same bytes as of 17 Sep 2026. |
| Same file in the newest published release | https://github.com/dres-dev/DRES/blob/2.0.4/doc/oas-client.json | release `2.0.4` (3 Jun 2024). Every path and request schema we use is identical; the only client-visible change in 2.0.5 is that `ApiClientTaskTemplateInfo.duration` became optional. |
| Full API specification (admin endpoints, used for the rehearsal setup) | `doc/oas.json` at the same tags | 2.0.4 and 2.0.5 |
| Old v2.0.0 draft with a GET submission endpoint | `doc/legacy.oas.json` | last changed 27 Mar 2023; no handler for it exists in any 2.0.x release |
| Backend sources for behaviour the specification leaves open | `backend/src/main/kotlin/dev/dres/` at `bd600c8d` | `api/rest/RestApi.kt`, `api/rest/AccessManager.kt`, `utilities/extensions/ContextExtensions.kt`, `api/rest/handler/{system,submission,log,evaluation/client}/*`, `api/rest/types/evaluation/submission/ApiClient*.kt`, `run/InteractiveSynchronousRunManager.kt`, `run/validation/*`, `run/transformer/*` |
| DRES wiki | https://github.com/dres-dev/DRES/wiki (wiki repository commit `9a82c507`, 24 Jul 2026) | pages Building a Client, Submissions, Interaction Logging, Setup Guide, Run Management, Scoring, Judgement |
| README (legacy submission notes, logging format pointer) | https://github.com/dres-dev/DRES/blob/v2.0.5/README.md | the README still mentions `/api/v1/submit`, but the 2.0.x backends no longer register it |
| Client examples | https://github.com/dres-dev/Client-Examples (last commit `f2135776`, 22 Jan 2024) | Java, Kotlin, C#, TypeScript clients generated from `oas-client.json`; no Python example |

The DRES README points to an Overleaf working document for the VBS logging format
(https://www.overleaf.com/read/rppygxshvhrn). We did not read it: it needs a browser session. The
message shapes we send come from the OpenAPI specification, which is what DRES actually parses.

## 2. The participant flow at VBS

**Accounts and login.** The organisers give each team one or more `PARTICIPANT` accounts.
`POST /api/v2/login` with `{"username", "password"}` returns an `ApiUser` whose `sessionId` is the
session token. DRES accepts the token either as the `SESSIONID` cookie or as the `session` query
parameter, and the query parameter wins when both are present. DRES re-issues the cookie with the
`Secure` flag on every request, so a normal cookie jar stops sending it over plain HTTP. The client
therefore sends `?session=<token>` on every call and never depends on cookies (checked live, see
section 5). A token lives until logout or a DRES restart; a request with a dead token gets HTTP 401.

**Choosing the evaluation.** `GET /api/v2/client/evaluation/list` returns every evaluation whose
template contains the user's team, not only the running ones (our test showed `CREATED` and
`ACTIVE` evaluations side by side), as `ApiClientEvaluationInfo` (`id`, `name`, `type`, `status`, `templateId`, `teams`, `taskTemplates`).
All later calls take that `id` as `{evaluationId}`. VBS runs are `SYNCHRONOUS`: the operator
shows the same task to all teams at the same time. The UI should preselect the single `ACTIVE`
evaluation and let the operator switch when there are several.

**Task lifecycle.** The operator starts the evaluation, selects a task and starts it. DRES shows a
short countdown (on 2.0.4, `timeElapsed` was -4 s right after the start), then the task runs for its
duration and ends on time or when the operator aborts it.

* `GET /api/v2/client/evaluation/currentTask/{evaluationId}` gives the task's `name`, `taskGroup`,
  `taskType` and `duration` in seconds. It answers 404 before the first task is selected (the client
  returns `None`), and after a task ends it keeps returning that ended task until the next one is
  selected. It does not say whether the task is running.
* `GET /api/v2/evaluation/{evaluationId}/state` gives `taskStatus` (`NO_TASK`, `CREATED`,
  `PREPARING`, `RUNNING`, `ENDED`, `IGNORED`), `timeLeft` and `timeElapsed` in seconds. Use it
  for the running/ended indicator and the countdown. Poll it about once a second: DRES viewers poll at
  that rate too.

The task text or video clip is shown on the DRES viewer (the projector at VBS). The client
specification has no endpoint for the query hint.

**Submitting.** `POST /api/v2/submit/{evaluationId}` with an `ApiClientSubmission`:

```json
{"answerSets": [{"answers": [{"mediaItemName": "00017", "mediaItemCollectionName": "V3C",
                              "start": 431000, "end": 431000}]}]}
```

* An answer is temporal when it has `mediaItemName`, `start` and `end` (milliseconds), an item answer
  when it has only `mediaItemName`, and a text answer when it has only `text`.
* `mediaItemCollectionName` is optional; without it the item name must be unique across all
  collections loaded in DRES. When a collection is imported with `collection scan`, each item is named
  after its file name without the extension (for V3C, the 5-digit video id).
* `taskId` and `taskName` can be left out. In synchronous runs DRES overwrites `taskId` with the running
  task anyway.
* Every answer set gets its own verdict. The judged validator (AVS) looks only at the first answer of
  a set, and the temporal validator (KIS) needs every answer of a set to be correct. So KIS and text
  answers go as one set with one answer, and an AVS submission sends one set per shot.

**Verdicts.** HTTP 200 carries `{"status": true, "submission": "CORRECT"|"WRONG"|"UNDECIDABLE",
"description"}`. When a submission has several answer sets, the verdict shown is the most frequent one.
HTTP 202 carries `INDETERMINATE`: the answer waits for a human judge (AVS, and QA tasks with a
`JUDGEMENT` target). The client specification has no endpoint for the judge's later decision; only
the DRES viewer and scoreboard show it.

| Situation | HTTP | `description` (DRES 2.0.4, observed) |
|---|---|---|
| Right segment | 200 | `Submission correct, well done!` |
| Wrong segment or text | 200 | `Submission wrong, try again!` |
| Waiting for a judge | 202 | `Submission received, awaiting verdict!` |
| Same answer again | 412 | `Submission <id> was rejected by filter: Duplicate submission received.` |
| Correct already found (KIS limit 1) | 412 | `... Maximum number of correct submissions (1) exceeded for the team.` |
| Operator started the evaluation but no task yet | 500 | `Could not find ongoing task in run manager.` |
| Task has ended | 500 | `Task run '<name>' is currently not running.` |
| Unknown media item | 500 | `Could not find media item with name '99999'. Maybe collection name is required.` |
| Wrong password | 401 | `Unauthorized request!` (2.0.4 replaces the handler's own message) |
| Dead or missing session | 401 | empty body |

The client raises `DresError` for every non-2xx answer, with `status` and the server's `description`
(session token and password removed). The UI should show the description as it is.

**Scoring.** From the DRES wiki, not checked in the code: a KIS correct submission earns between 1000
and 500 points depending on how quickly it arrives, and every wrong submission before it costs 100.
AVS rewards each distinct correct video and takes 0.2 of a video's value for each wrong submission on
it. So wrong KIS submissions cost points: do not guess blindly.

**Two rules that decide KIS and AVS verdicts** (from the validators, the KIS rule confirmed live):

1. *KIS: containment.* A temporal answer is correct only if `[start, end]` lies completely inside the
   target segment. With a 20–30 s target, a 18–22 s window and a 28–32 s window were both judged WRONG,
   19.999 s and 30.001 s points were WRONG, and a 25 s point was CORRECT. A ±2 s window around a frame
   that is inside the target can therefore still be wrong near the boundary. **Submit a point
   (`start_ms == end_ms`) at the chosen frame time**, or a window of at most a few hundred ms.
2. *AVS: one shot per answer.* The AVS task type preset has the `MAP_TO_SEGMENT` option: DRES widens
   each answer to the predefined shot that contains it, and refuses (HTTP 500, "range does not fall
   within one segment") an answer whose start and end lie in different shots. A point answer always
   stays inside one shot.

**Logging.** Two write-only endpoints. Both require the `session` query parameter and a
`PARTICIPANT` or `ADMIN` session, and neither needs a running task.

* `POST /api/v2/log/query/{evaluationId}` with `QueryEventLog` `{timestamp, events}`. Each
  `QueryEvent` has `timestamp` (Unix ms), `category` (`TEXT`, `IMAGE`, `SKETCH`, `FILTER`,
  `BROWSING`, `COOPERATION`, `OTHER`), `type` (free string) and `value` (string). All four fields
  are required.
* `POST /api/v2/log/result/{evaluationId}` with `QueryResultLog` `{timestamp, sortType,
  resultSetAvailability, results, events}`. `results` is a list of `RankedAnswer` `{answer, rank}`
  where `answer` has the same shape as a submission answer. All five fields are required. `sortType`
  and `resultSetAvailability` are free strings in the specification.

DRES adds its server time to each log and passes it to its event stream, which is written to disk
for analysis after the event. DRES barely checks the content: 2.0.4 answered 200 to an event with an
extra field, an event without `value`, a log with `timestamp_ms` instead of `timestamp`, and a
result whose answer used `mediaItem` instead of `mediaItemName`. Only an unknown category or a
non-string `value` gave HTTP 500. Such mistakes would be lost without any error, so the client checks
every shape before sending. Our conventions:
`query_event("TEXT", "jointEmbedding", <scene text>)` per scene, `FILTER` for video filters,
`IMAGE`/`similarImage` for more-like-this, `BROWSING` for result-list and video-view actions; result
logs use `sortType="score"`, `resultSetAvailability="top"`, ranks from 1, and the answers exactly as
they would be submitted. Check the VBS 2027 participant instructions for any required event types or
result-list length.

## 3. Endpoints used by the client

| Client method | Method and path | Request body | 2xx response |
|---|---|---|---|
| `login(user, password)` | `POST /api/v2/login` | `LoginRequest` | `ApiUser` |
| `logout()` | `GET /api/v2/logout` | – | `SuccessStatus` |
| `evaluations()` | `GET /api/v2/client/evaluation/list` | – | `ApiClientEvaluationInfo[]` |
| `task_info(evaluation_id)` | `GET /api/v2/client/evaluation/currentTask/{evaluationId}` | – | `ApiClientTaskTemplateInfo`, or `None` on 404 "no active task" |
| `evaluation_state(evaluation_id)` | `GET /api/v2/evaluation/{evaluationId}/state` | – | `ApiEvaluationState` |
| `submit_kis`, `submit_text`, `submit_avs`, `submit` | `POST /api/v2/submit/{evaluationId}` | `ApiClientSubmission` | `SuccessfulSubmissionsStatus` (200 or 202) |
| `log_query(evaluation_id, events)` | `POST /api/v2/log/query/{evaluationId}` | `QueryEventLog` | `SuccessStatus` |
| `log_results(evaluation_id, result_log)` | `POST /api/v2/log/result/{evaluationId}` | `QueryResultLog` | `SuccessStatus` |

Every call except `login` carries `?session=<token>`. There is no legacy fallback: no 2.0.x
backend registers the old GET submission handler, and neither `oas-client.json` nor `oas.json`
lists one. Timeouts default to 10 s. Nothing is retried, because a repeated submission would count
twice.

**Secrets.** The client never logs the password or the token and removes both from error messages.
Two places outside our code still see the token: the DRES server writes every request's query
parameters, including `session=`, to its `logs/rest.log`; and `urllib3` prints full request URLs if
someone turns on DEBUG logging for the root logger. Keep `urllib3` at WARNING on the competition laptop.

## 4. Tests

`python -m pytest tests/test_dres.py -q` (no network). The mock server implements the eight
paths above and validates every request body, and its own responses, against the schema fragments
embedded in the test file. The fragments are copied verbatim from `oas-client.json` at `bd600c8d`.
Setting `DRES_CLIENT_SPEC=<path to oas-client.json>` also compares those fragments, the
response status codes and the request schemas with a downloaded specification. Run that check again
when the organisers announce the DRES version for VBS 2027.

The tests cover login success and failure, session propagation through the query parameter (the
mock ignores cookies, as a Secure cookie would be ignored over HTTP), calls before login, logout and
re-login, expired sessions, the evaluation list, task info in all three cases, evaluation state, the
payloads for KIS (with and without collection), text, AVS (one set per shot) and a generic
multi-answer set, both log payloads, errors with JSON and non-JSON bodies, redaction of the token,
timeouts, an unreachable server, argument validation before any request, and the absence of secrets
in logs.

## 5. What was verified against a real DRES

Done on 17 Sep 2026 on the team laptop (Windows 11, no admin rights, OpenJDK Temurin 25.0.2 already
installed; Docker was installed but its daemon was not running and was not needed).

* Release `dres-dist-2.0.4.tar` (230 MB) from https://github.com/dres-dev/DRES/releases/tag/2.0.4:
  download 4 min 17 s, extract under 1 s, backend ready in under 20 s. The 2.0.5 tag has no
  published distribution and building from source needs Gradle and npm, so 2.0.4 was used. Its
  client specification matches 2.0.5 for everything the client sends.
* The bundled `lib/ffmpeg` binaries are for Linux. On Windows, creating an evaluation fails with
  "Required media file could not be prepared" until `cache.ffmpegPath` points to a Windows FFmpeg
  directory. Creating an evaluation also fails if the media files are missing on disk.
* Setup through the admin REST API: 2 users created from the CLI (piped into stdin), a collection
  with two 120 s videos, a template with a Textual KIS task (target 20–30 s of item `00001`), an AVS
  task (JUDGEMENT target, preset without `MAP_TO_SEGMENT` because no shot segments were imported)
  and a QA task (TEXT target), one team, and SYNCHRONOUS evaluations.

Results with `DresClient` as the participant:

| Check | Result |
|---|---|
| Login with wrong password | `DresError` 401 "Unauthorized request!" |
| Login | `ApiUser` with role `PARTICIPANT` |
| Evaluation list | all evaluations of the team, `CREATED` and `ACTIVE`, with their task template names |
| `task_info` before any task | `None` |
| `task_info` with an unknown id | `DresError` 404 "Evaluation no-such-evaluation not found." |
| `evaluation_state` during a task | `taskStatus` RUNNING, `timeLeft` 304, `timeElapsed` -4 right after the start |
| `submit_kis` wrong segment, then again | WRONG, then 412 duplicate |
| `submit_kis` unknown item | 500 "Could not find media item with name '99999' ..." |
| `submit_kis` inside the target, with collection name | CORRECT |
| Containment rule | 18–22 s WRONG, 28–32 s WRONG, 19.999 s WRONG, 30.001 s WRONG, 25 s CORRECT, then 412 limit |
| `log_query` (TEXT and BROWSING events), `log_results` (2 ranked answers) | `SuccessStatus` "Log received" / "Log entry received." |
| `submit_avs` with two shots | 202 INDETERMINATE; the admin API shows one submission with two answer sets, each TEMPORAL and INDETERMINATE |
| `submit_text` "red", then " blue " | WRONG, then CORRECT (the client trims whitespace) |
| Malformed logs sent with raw requests | 200 for an extra field, a missing `value`, `timestamp_ms` instead of `timestamp`, and a `mediaItem` answer field; 500 for category `QUERY` and for an object `value` |
| Submit before a task / after the task ended | 500 "Could not find ongoing task in run manager." / 500 "Task run '...' is currently not running." |
| `task_info` after the task ended | still returns the ended task |
| Client with all cookies blocked | login, evaluation list and `log_query` work: the query parameter is enough |
| Old token after logout | 401 |
| Closed port, 2 s timeout | `DresError` "timed out after 2 s" (on Windows a refused localhost connection shows up as a connect timeout) |

Not verified live: `MAP_TO_SEGMENT` with imported shot segments, judge verdicts reaching the scoreboard,
HTTPS, a reverse proxy, and ASYNCHRONOUS runs.

## 6. Rehearsal checklist

Before VBS, repeat this with the real tool (web UI → `server.py` → `DresClient`), not only the client.

**Start a local DRES (about 10 minutes, no admin rights)**

1. Java 17 or newer (`java -version`). The 2.0.4 build ran on Java 25.
2. Download `dres-dist-2.0.4.tar` (or the version the organisers announce) from the DRES releases
   page and extract it. Create a working directory with a `config.json`:

   ```json
   {"httpPort": 8080, "enableSsl": false, "dataPath": "./data", "externalMediaLocation": "./external",
    "cache": {"cachePath": "./cache", "ffmpegPath": "C:/path/to/ffmpeg/bin"}}
   ```

3. From that directory: `java -classpath ".;<dres-dist>/lib/*" dev.dres.DRES config.json` on Windows
   (`:` instead of `;` on Linux/macOS), or `<dres-dist>/bin/backend config.json`.
4. In the `DRES>` prompt: `user create -u admin -p <pw> -r ADMIN` and
   `user create -u team1 -p <pw> -r PARTICIPANT` (plus a `JUDGE` user to rehearse AVS).
5. Put a few videos from the collection we will search into a media folder. In the web UI at
   http://localhost:8080 (or with the CLI `collection create` / `collection scan`), create the
   collection under **the same name and with the same item names as our `collection.json` and
   `metadata.json`**. Import shot segments if AVS will use `MAP_TO_SEGMENT`.
6. Build a template with the VBS presets (Textual KIS, Visual KIS, Ad-hoc Video Search, and a QA task
   with a TEXT or JUDGEMENT target), add the team with `team1`, create a SYNCHRONOUS evaluation, start it.

**Run through the tool**

- [ ] Log in from the UI; the status pill shows the user; `/api/status` reports `connected`.
- [ ] The evaluation selector lists the evaluation; choose it.
- [ ] Before a task starts: the UI shows "no active task" and does not crash.
- [ ] Start a KIS task: the task name and countdown match the DRES viewer to within about 1 s.
- [ ] Submit a frame inside the target → CORRECT. Submit one just outside → WRONG. Submit the same
      answer twice → 412 shown as a rejection, not as a crash.
- [ ] Check which segment the tool sends: a point or a very short range (see the containment rule).
- [ ] Submit a media item from another collection or with a typo → the 500 message appears in the
      submission log.
- [ ] AVS: send a basket of 5 shots → 202 pending. The judge rates them in the DRES judgement view,
      and the scoreboard changes.
- [ ] QA: submit a text answer → verdict (or pending for JUDGEMENT targets).
- [ ] After the task ends: submissions are refused with a readable message; the UI shows the task as ended
      (use `evaluation_state`, not `task_info`).
- [ ] Logs: after a few searches, confirm that DRES accepted `log/query` and `log/result` (tool log
      shows `dres_sent`), and that `log_dir/*.jsonl` holds the same content.
- [ ] Restart DRES during a task: the next call returns 401 and the UI asks to log in again.
- [ ] Unplug the network: calls fail within the timeout with a clear message, and the UI stays usable.
- [ ] Nothing in the tool's console or `log_dir` contains the password or the session token
      (`grep` for both).

**On the day**

- [ ] Get the DRES URL, the accounts, the collection name and the media item naming from the organisers.
      Log in once during the test session.
- [ ] Run `DRES_CLIENT_SPEC=<downloaded oas-client.json> python -m pytest tests/test_dres.py -q`
      against the specification of the DRES version in use.
- [ ] Check the laptop clock: log timestamps use client time.
- [ ] Keep `urllib3` logging at WARNING.

## 7. Open questions for the organisers

- The DRES version and base URL for VBS 2027, and whether it runs behind HTTPS or a proxy.
- The collection names (`mediaItemCollectionName`) and item naming for V3C, Marine Video Kit and
  GynSurg. Is the collection name required?
- Required log content: mandatory event categories and types, the expected length of result lists, and
  the values for `sortType` and `resultSetAvailability`.
- Whether AVS and QA tasks use `MAP_TO_SEGMENT`, JUDGEMENT or TEXT targets, and the submission limits
  per task.
