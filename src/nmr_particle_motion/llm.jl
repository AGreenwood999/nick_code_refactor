# combine_py_files_for_llm.jl
using Glob

function combine_for_llm(output_file="combined_for_llm.md"; recursive=false)
    pattern = recursive ? "**/*.py" : "*.py"
    files = glob(pattern, pwd())

    open(output_file, "w") do io
        write(io, "# Combined Python codebase\n\n")
        write(io, "Total files: $(length(files))\n")
        write(io, "Base directory: $(pwd())\n\n")

        for f in sort(files)   # alphabetical order helps
            rel = relpath(f, pwd())

            write(io, "## FILE: $(rel)\n\n")
            write(io, "```python\n")

            try
                content = read(f, String)
                # Escape triple backticks just in case
                content = replace(content, "```" => "\\`\\`\\`")
                write(io, content)
            catch e
                write(io, "# Error reading file: $e")
            end

            write(io, "\n```\n\n")
        end
    end

    println("Created Markdown file for LLM: $output_file")
    println("Found and included $(length(files)) .py files")
end

# Usage
combine_for_llm(recursive=false)
# combine_for_llm("whole_project.md", recursive=true)
