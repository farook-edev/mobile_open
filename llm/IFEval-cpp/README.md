# IFEval C++ implementation

This Project is a modified version of MLCommons' C++ IFEval evaluation code, designed to run standalone using previously written responses stored in `json` format.

## Operation

This tool is a bit more manual than the python implementation released by google, the primary difference is that it operates on a single `json` file rather than 2 separate `jsonl` files, and produces a single result set containing both loose and strict parameters. 

### Tools
Because of the differences above, the input data from the python implementation will need to be processed, and so will the output. The following are the tools that will be used for that:
- `merger.py`: used to merge the 2 input files (`input_data.jsonl` and `input_response_data_gpt4_20231107_145030.jsonl`).
- `jsoner.py`: used to convert `merged.jsonl` to `merged.json` for use with the C++ tool
- `33merger.py`: used to add missing field to the `IFEval33` result files. (uses an existing merged `jsonl` file)
- `process-cpp.py`: used to convert the merged output from the C++ code into 2 separate `loose` and `strict` files that are in compatible format with the python code.

### Building and Running
Compiling the code is simple, a C++17 or later compiler and `make` should handle everything.

Once the code is compiled, and `merged.json` file is created using `merger.py` and `jsoner.py`, the command to run should simply be `cat merged.json | main > cpp-results.txt`.

If everything runs without issue, the file should then be used with `process-cpp.py` to generate the loose and strict result files for comparison.

## Important Notes
This tool was initially designed for internal testing only, so it might contain commented code or quick patchwork fixes.
