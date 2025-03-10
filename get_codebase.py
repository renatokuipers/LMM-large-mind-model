import os
from os import system
import sys

def get_user_input():
    directory = input("Enter the path of the directory to analyze: ").strip()
    if not os.path.isdir(directory):
        sys.exit(1)
    return os.path.abspath(directory)

def should_exclude_dir(dirname):
    excluded_dirs = {'node_modules', '.git', '__pycache__', '.venv', 'venv'}
    return dirname in excluded_dirs

def should_include_file(filename):
    included_extensions = {'.js', '.jsx', '.json', '.css', '.html', '.py', '.env', '.tsx', '.ts', '.yaml', '.txt', '.yml', '.env'}
    _, ext = os.path.splitext(filename)
    return ext in included_extensions

def build_tree(startpath):
    tree_lines = []
    prefix_stack = []

    for root, dirs, files in os.walk(startpath):
        dirs.sort()
        files.sort()
        dirs[:] = [d for d in dirs if not should_exclude_dir(d)]
        relative_path = os.path.relpath(root, startpath)
        level = 0 if relative_path == '.' else relative_path.count(os.sep) + 1
        while len(prefix_stack) < level:
            prefix_stack.append('│   ')
        while len(prefix_stack) > level:
            prefix_stack.pop()
        is_last = is_last_item(root, startpath)
        connector = '└── ' if is_last else '├── '
        indent = ''.join(prefix_stack[:-1])
        if level == 0:
            tree_lines.append(f'{os.path.basename(startpath)}/')
        else:
            tree_lines.append(f"{indent}{connector}{os.path.basename(root)}/")
        if level > 0:
            prefix_stack[-1] = '    ' if is_last else '│   '
        relevant_files = [f for f in files if should_include_file(f)]
        for i, file in enumerate(relevant_files):
            file_is_last = i == len(relevant_files) - 1
            file_connector = '└── ' if file_is_last else '├── '
            file_indent = ''.join(prefix_stack)
            tree_lines.append(f"{file_indent}{file_connector}{file}")

    return tree_lines

def is_last_item(current_path, startpath):
    parent = os.path.dirname(current_path)
    if parent == current_path:
        return True
    try:
        items = [d for d in os.listdir(parent) if os.path.isdir(os.path.join(parent, d)) and not should_exclude_dir(d) and not "test_split.txt"]
        return items[-1] == os.path.basename(current_path)
    except IndexError:
        return False

def collect_files(startpath):
    collected_files = []
    for root, dirs, files in os.walk(startpath):
        dirs[:] = [d for d in dirs if not should_exclude_dir(d)]
        for file in files:
            if should_include_file(file):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except (UnicodeDecodeError, PermissionError) as e:
                    content = "<Could not read file>"
                    print(f"Warning: Could not read file '{file_path}': {e}")
                relative_path = os.path.relpath(file_path, startpath)
                collected_files.append((relative_path, content))
    return collected_files

def write_project_layout(tree_lines, files_contents, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("#######################\n")
        f.write("Project Layout:\n\n")
        for line in tree_lines:
            f.write(line + '\n')
        f.write("\n#######################\n\n")
        f.write("Codebase:\n#######################\n")
        for filename, content in files_contents:
            f.write(f"#{filename}#\n")
            f.write("#######################\n\n")
            f.write(content + "\n\n")
            f.write("#######################\n\n")

def main():
    system("cls")
    system("del project_layout.txt")
    startpath = get_user_input()
    tree_lines = build_tree(startpath)
    files_contents = collect_files(startpath)
    output_file = "project_layout.txt"
    write_project_layout(tree_lines, files_contents, output_file)

if __name__ == "__main__":
    main()
