import os
import subprocess
import requests
import json
from getpass import getpass
import sys

def run_command(command, cwd):
    """Runs a command and handles errors."""
    try:
        print(f"  > Executing: {' '.join(command)}")
        # Use shell=True for Windows compatibility if git is not in PATH
        is_windows = sys.platform.startswith('win')
        subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True, shell=is_windows)
    except subprocess.CalledProcessError as e:
        print("\n--- ERROR ---")
        print(f"Command failed: {' '.join(command)}")
        print("STDERR:", e.stderr)
        print("-------------")
        sys.exit(1)

def create_github_repo(username, repo_name, token):
    """Creates a new repository on GitHub."""
    print(f"\n[Step 1/3] Creating GitHub repository '{username}/{repo_name}'...")
    url = "https://api.github.com/user/repos"
    headers = {"Authorization": f"token {token}"}
    data = {"name": repo_name, "description": "My catchment foundation model project.", "private": False}
    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    if response.status_code == 201:
        print(f"  > Repository '{repo_name}' created successfully.")
        return response.json()["clone_url"]
    elif response.status_code == 422 and "name already exists" in response.text:
        print(f"  > Repository '{repo_name}' already exists on GitHub. Will use it.")
        return f"https://github.com/{username}/{repo_name}.git"
    else:
        print("\n--- ERROR ---")
        print(f"Failed to create GitHub repository. Status: {response.status_code}, Response: {response.json()}")
        print("Please check your username and personal access token (it needs the 'repo' scope).")
        sys.exit(1)

def main():
    print("--- Push Current Directory to a New GitHub Repo ---")
    current_dir = os.getcwd()
    
    if os.path.isdir(".git"):
        print("\nWarning: This directory is already a Git repository.")
        if input("Do you want to continue and push to a new remote? (y/n): ").lower() != 'y':
            sys.exit("Operation cancelled.")
    
    # Get user input
    github_username = input("Enter your GitHub username: ")
    repo_name = input("Enter the name for the new GitHub repository: ")
    github_token = getpass("Enter your GitHub Personal Access Token (with 'repo' scope): ")

    # 1. Create the repository on GitHub
    clone_url = create_github_repo(github_username, repo_name, github_token)

    # 2. Initialize Git locally if needed
    print("\n[Step 2/3] Initializing local Git repository...")
    if not os.path.isdir(".git"):
        run_command(["git", "init", "-b", "main"], cwd=current_dir)
        print("  > Initialized a new Git repository in the current directory.")

    # 3. Add, commit, and push all files
    print("\n[Step 3/3] Adding, committing, and pushing all files...")
    
    # Add a .gitignore to avoid pushing large/unnecessary files
    gitignore_path = os.path.join(current_dir, ".gitignore")
    if not os.path.exists(gitignore_path):
        print("  > Creating a '.gitignore' file to exclude common temporary files.")
        with open(gitignore_path, "w") as f:
            f.write("__pycache__/\n")
            f.write("*.pyc\n")
            f.write(".env\n")
            f.write(".venv\n")
            f.write("*.pth\n") # Exclude trained models
            f.write("*.tif\n")  # Exclude data files
            f.write("*.tiff\n")
            f.write("catchment-foundation-model-ui/\n") # Exclude the folder created by the previous script

    run_command(["git", "add", "."], cwd=current_dir)
    
    # Use --allow-empty-message for simplicity in case of no changes
    # And check if there is anything to commit
    status_output = subprocess.run(["git", "status", "--porcelain"], cwd=current_dir, capture_output=True, text=True).stdout
    if status_output:
        run_command(["git", "commit", "-m", "Initial project upload"], cwd=current_dir)
    else:
        print("  > No new files to commit.")
        
    # Check for existing remote and set it
    remotes = subprocess.run(["git", "remote"], cwd=current_dir, capture_output=True, text=True).stdout
    if "origin" in remotes.split():
        print("  > Git remote 'origin' already exists. Updating its URL.")
        run_command(["git", "remote", "set-url", "origin", clone_url], cwd=current_dir)
    else:
        print("  > Adding new git remote 'origin'.")
        run_command(["git", "remote", "add", "origin", clone_url], cwd=current_dir)

    # Push to GitHub
    print("  > Pushing files to GitHub...")
    run_command(["git", "push", "-u", "origin", "main"], cwd=current_dir)

    print("\n--- All Done! ---")
    print("Your entire project directory has been pushed to your new GitHub repository.")
    print(f"View it online at: {clone_url.replace('.git', '')}")

if __name__ == "__main__":
    main()
