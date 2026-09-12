# Contributing to DuckDo

Notes for working on DuckDo itself. The user-facing documentation is the
[README](README.md) and the pages under [docs/](docs/), which are published at
https://maxdemarzi.github.io/DuckDo/. The planning document is [dev/ROADMAP.md](dev/ROADMAP.md),
and [dev/UPDATING.md](dev/UPDATING.md) covers moving to a new DuckDB release.

## Testing

```sh
./build/release/test/unittest "test/*"
```

```sh
# with foundation models as well
DUCKDO_MODEL_DIR=$(pwd)/build/models ./build/release/test/unittest "test/*"
```

831 assertions across 30 test files, with three further files that run only when
`DUCKDO_MODEL_DIR` points at exported weights. They cover estimator recovery, diagnostics, error paths, guardrails, graph identification, discovery, mediation,
time-varying treatment, survival, the `do()` surface and end-to-end inference for both models.

Three further dev-only harnesses, none shipped:

```sh
python scripts/crosscheck_econml.py    # grades the estimators against EconML and DoWhy on IHDP
python scripts/coverage_check.py       # measures do_cate's empirical interval coverage
python scripts/benchmark.py            # seconds and peak memory, with a gate on both
python scripts/check_docs.py           # every SQL block in the docs still runs
python scripts/check_functions.py      # the function reference matches the extension
```

`check_docs.py` executes every block in the tutorial and the worked examples, in order,
against a fresh database — those two documents quote real output, and quoting real output
is worth nothing if the queries above it have stopped working. The README, assumptions
guide and function reference use illustrative fragments against tables that do not exist,
so those are parsed rather than executed, which catches a malformed query but not a stale
column name.

CI builds on Linux, macOS, Windows and WebAssembly, and checks more than a Windows build can
see. Linux compiles as C++11, where a struct with a default member initialiser is not an
aggregate and cannot be brace-initialised; macOS and WebAssembly make `idx_t` and `size_t`
different types, so `std::min(a, b)` across the two needs `std::min<idx_t>`. Both are worth
checking before a push, for instance with `pip install ziglang` and
`python -m ziglang c++ -target x86_64-linux-gnu -std=c++11 -fsyntax-only`, or `-target
aarch64-macos`. Tests run again under AddressSanitizer and UBSan on a debug build, where DuckDB
also asserts things a release build allows. Test files are formatted by DuckDB's own formatter,
which wants each file's group to be its directory: call `format_file_content` from
`duckdb/scripts/format_test_benchmark.py`.

CI also runs DuckDB's `format` and `tidy` checks, as
`python3 duckdb/scripts/format.py --all --check --directories src test`. Run **that**, not
clang-format by hand: a hand-run `clang-format --style=file` on a copy of the file is a weaker
check that accepts things CI rejects, which has cost a red build — it tolerated a short lambda on
one line where `AllowShortFunctionsOnASingleLine: false` requires three.

On Windows `format.py` needs a detour. `.clang-format` and `.clang-tidy` in this repo are symlinks
into `duckdb/`, and git checks them out as plain text files there, so clang-format reads the
*path* as if it were the config and fails with "YAML:1:1: error: not a mapping". Copy
`duckdb/.clang-format` over the root stub, run the check, and put the stub back:

```sh
pip install cmake-format "black==24.*"     # format.py wants both
cp .clang-format /tmp/stub && cp duckdb/.clang-format .clang-format
python duckdb/scripts/format.py --check --directories src test
cp /tmp/stub .clang-format
```

The pinned clang-format is `clang_format==11.0.1` — later versions disagree about line breaks and
will produce a diff CI rejects.

## Submitting to community extensions

[description.yml](description.yml) is ready to copy into a fork of
`duckdb/community-extensions`; update `version` and `ref` to the release commit first. Its
`hello_world` deliberately needs no download - a first impression that requires a 300 MB fetch is a
first impression most people never have.

## Building
### Managing dependencies
DuckDo currently has **no external dependencies** - the numerics are hand-rolled in `src/linalg.cpp`,
so `vcpkg.json` lists nothing and you can skip vcpkg entirely. Building the foundation-model
path with `-DDUCKDO_WITH_ONNX=ON` fetches a pinned ONNX Runtime release and stages it for you.

### Build steps
Now to build the extension, run:
```sh
make
```
The main binaries that will be built are:
```sh
./build/release/duckdb
./build/release/test/unittest
./build/release/extension/duckdo/duckdo.duckdb_extension
```
- `duckdb` is the binary for the duckdb shell with the extension code automatically loaded.
- `unittest` is the test runner of duckdb. Again, the extension is already linked into the binary.
- `duckdo.duckdb_extension` is the loadable binary as it would be distributed.

## Running the tests
Different tests can be created for DuckDB extensions. The primary way of testing DuckDB extensions should be the SQL tests in `./test/sql`. These SQL tests can be run using:
```sh
make test
```

### Installing the deployed binaries
To install your extension binaries from S3, you will need to do two things. Firstly, DuckDB should be launched with the
`allow_unsigned_extensions` option set to true. How to set this will depend on the client you're using. Some examples:

CLI:
```shell
duckdb -unsigned
```

Python:
```python
con = duckdb.connect(':memory:', config={'allow_unsigned_extensions' : 'true'})
```

NodeJS:
```js
db = new duckdb.Database(':memory:', {"allow_unsigned_extensions": "true"});
```

Secondly, you will need to set the repository endpoint in DuckDB to the HTTP url of your bucket + version of the extension
you want to install. To do this run the following SQL query in DuckDB:
```sql
SET custom_extension_repository='bucket.s3.eu-west-1.amazonaws.com/<your_extension_name>/latest';
```
Note that the `/latest` path will allow you to install the latest extension version available for your current version of
DuckDB. To specify a specific version, you can pass the version instead.

After running these steps, you can install and load your extension using the regular INSTALL/LOAD commands in DuckDB:
```sql
INSTALL duckdo;
LOAD duckdo;
```

## Setting up CLion

### Opening project
Configuring CLion with this extension requires a little work. Firstly, make sure that the DuckDB submodule is available.
Then make sure to open `./duckdb/CMakeLists.txt` (so not the top level `CMakeLists.txt` file from this repo) as a project in CLion.
Now to fix your project path go to `tools->CMake->Change Project Root`([docs](https://www.jetbrains.com/help/clion/change-project-root-directory.html)) to set the project root to the root dir of this repo.

### Debugging
To set up debugging in CLion, there are two simple steps required. Firstly, in `CLion -> Settings / Preferences -> Build, Execution, Deploy -> CMake` you will need to add the desired builds (e.g. Debug, Release, RelDebug, etc). There's different ways to configure this, but the easiest is to leave all empty, except the `build path`, which needs to be set to `../build/{build type}`, and CMake Options to which the following flag should be added, with the path to the extension CMakeList:

```
-DDUCKDB_EXTENSION_CONFIGS=<path_to_the_exentension_CMakeLists.txt>
```

The second step is to configure the unittest runner as a run/debug configuration. To do this, go to `Run -> Edit Configurations` and click `+ -> Cmake Application`. The target and executable should be `unittest`. This will run all the DuckDB tests. To specify only running the extension specific tests, add `--test-dir ../../.. [sql]` to the `Program Arguments`. Note that it is recommended to use the `unittest` executable for testing/development within CLion. The actual DuckDB CLI currently does not reliably work as a run target in CLion.
