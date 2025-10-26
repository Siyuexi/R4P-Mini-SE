import uuid
import os
import subprocess
import re

class WorkSpace():

    def __init__(self, root_dir, temp_dir, instance_id):
        self.uuid = str(uuid.uuid4())[:8]
        self.project = '-'.join(instance_id.split('-')[:-1])
        self.src_path = f'{root_dir}/project/{self.project}'
        self.path = f'{temp_dir}/temp_{instance_id}_{self.uuid}'
        self.instance_id = instance_id

    def init_src(self):
        project_name = self.project.replace('__', '/')
        if not os.path.exists(self.src_path):
            subprocess.run(
                    ['git', 'clone', f'git@github.com:{project_name}.git', self.src_path],
                    check=True,
                    capture_output=True,
                    text=True
                )
    
    def update_src(self):
        if os.path.exists(self.src_path):
            subprocess.run(
                ['git', '-C', self.src_path, 'pull'],
                check=True,
                capture_output=True,
                text=True
            )

    def create_ws(self, base_commit, git=True):
        if os.path.exists(self.path):
            self.del_ws()
        os.mkdir(self.path)

        temp_archive_file = os.path.abspath(f'{self.path}/archive.tar')

        try:
            archive_cmd = [
                'git', '-C', self.src_path,
                'archive', '--format=tar', f'--output={temp_archive_file}',
                base_commit
            ]
            subprocess.run(
                archive_cmd,
                check=True,
                capture_output=True,
                text=True
            )

            extract_cmd = [
                'tar', '-x', '-f', temp_archive_file, '-C', self.path
            ]
            subprocess.run(
                extract_cmd,
                check=True,
                capture_output=True,
                text=True
            )
        
        except subprocess.CalledProcessError as e:
            error_msg = f"""
            ⚠️ Archive command failed!
            - Command: {' '.join(e.cmd)}
            - Exit code: {e.returncode}
            - Output: {e.stdout}
            - Error: {e.stderr}
            - Src path: {self.src_path}
            - Dest path: {self.path}
            - Instance ID: {self.instance_id}
            """
            print(error_msg)
            raise e
        
        finally:
            if os.path.exists(temp_archive_file):
                os.remove(temp_archive_file)

        if git:
            git_cmd = f"""
            cd {self.path} && 
            git init && 
            git add . && 
            git commit -m 'Initial commit'
            """
            subprocess.run(
                git_cmd,
                shell=True,
                executable='/bin/bash',
                check=True,
                capture_output=True,
                text=True
            )

    def del_ws(self):
        if os.path.exists(self.path):
            cmd = f"rm -rf {self.path}"
            os.system(cmd)

    def get_diff(self) -> str:
        if not os.path.isdir(self.path):
            return ""
        
        try:
            result = subprocess.run(
                ["git", "diff", "HEAD"],
                cwd=self.path,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                result = subprocess.run(
                    ["git", "diff"],
                    cwd=self.path,
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    return f"Error obtaining diff patch: {result.stderr}\nFind and fix the problem with `search_tool` and `edit_tool`, then use `patch_submission` again to generate and submit the patch."
            
            patch = result.stdout
            if patch.strip() == '':
                return "The patch is empty, meanning that you have not performed any valid edit yet. Please use `edit_tool` to fix buggy code before patch submission."
            return f"Successfully obtained and submitted diff patch:\n[PATCH]\n{patch}\n[/PATCH]\nReview this patch. If you find anything wrong, you can use `edit_tool` to fix them, and then use `patch_submission` again to generate a new one. Otherwise, you can just end this conversation."
        except Exception as e:
            return f"Error obtaining diff patch: {str(e)}\nFind and fix the problem with `search_tool` and `edit_tool`, then use `patch_submission` again to generate and submit the patch."
