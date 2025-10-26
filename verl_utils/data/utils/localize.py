import re
import json

"""
Localize the bug fix position in diff patch string.
Could be useful for process reward "fault localization".

Return a json string of a list of edits (file, func, line_start, line_end), e.g.,
[
    {'file': 'astropy/io/fits/card.py', 'func': 'fromstring', 'line_start': 547, 'line_end': 559},
    {'file': 'astropy/io/fits/header.py', 'func': 'fromstring', 'line_start': 329, 'line_end': 341},
    {'file': 'src/sqlfluff/cli/commands.py', 'func': 'dump_file_payload', 'line_start': 455, 'line_end': 461},
    {'file': 'src/sqlfluff/cli/commands.py', 'func': 'dump_file_payload', 'line_start': 509, 'line_end': 516},
    {'file': 'src/sqlfluff/cli/commands.py', 'func': 'do_fixes', 'line_start': 704, 'line_end': 710},
    {'file': 'astropy/modeling/separable.py', 'func': '_cstack', 'line_start': 242, 'line_end': 248}
]
"""

def get_hunk_location(patch_content):
    """
    WARNING: This method should only be used on golden patch. DO NOT use it on generated patch since the generated patch might not follow the standard unidiff format, causing incorrect function position extraction.
    """
    location_list = []
    file_pattern = re.compile(r'diff --git a/(.*?) b/(.*?)\n(.*?)(?=\ndiff --git |\Z)', re.MULTILINE | re.DOTALL)
    func_pattern = re.compile(r'@@ -(\d+),(\d+) \+(\d+),(\d+) @@ (?:.*?class.*?)?\s*(?:@.*?\n\s*)*(?:async\s+)?def ([a-zA-Z_][a-zA-Z0-9_]*)\(', re.MULTILINE)
    
    file_matches = file_pattern.finditer(patch_content)
    for match in file_matches:
        origin_file = match.group(1)
        file_patch_content = match.group(3)
        func_list = func_pattern.findall(file_patch_content)
        for func in func_list:
            origin_start_line = int(func[0])
            origin_end_line = int(func[0]) + int(func[1]) - 1
            func_name = func[4]
            location_list.append(
                {
                    'file': origin_file,
                    'func': func_name,
                    'line_start': origin_start_line,
                    'line_end': origin_end_line,
                }
            )
    
    return json.dumps(location_list, indent=2)

if __name__ == "__main__":
    with open('data/debug_patch_example.patch', 'r') as f:
        patch_content = f.read()
    
    location_str = get_hunk_location(patch_content)
    print(location_str)
    localtion_list = json.loads(location_str)
    for loc in localtion_list:
        print(loc)
