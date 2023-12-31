import os
import json
import git

from constants import (
    REPO_PATH,
    PERSIST_DIRECTORY,
)


def write_json_store(data):
    # Write data to a JSON file
    existing_data = load_json_store()
    existing_data.update(data)

    with open(f"{PERSIST_DIRECTORY}/data.json", "w") as file:
        json.dump(existing_data, file)

    return existing_data


def load_json_store():
    loaded_data = {}  # Initialize an empty dictionary
    # Read data from a JSON file
    try:
        with open(f"{PERSIST_DIRECTORY}/data.json", "r") as file:
            loaded_data.update(json.load(file))
    except FileNotFoundError:
        pass  # Handle the case when the file doesn't exist

    return loaded_data


def get_repo_latest_commit_by_branch(branch="master"):
    # Open the Git repository
    repo = git.Repo(REPO_PATH)

    # Get the latest commit on the "master" branch
    return repo.commit(branch)


def update_repo_current_embedded_commit():
    latest_commit = get_repo_latest_commit_by_branch()
    write_json_store({"embedded_commit": latest_commit.hexsha})

    return True


def diff_hash():
    latest_commit = get_repo_latest_commit_by_branch()
    json = load_json_store()
    print(json)
    # Show the changes between the latest commit and the stored commit
    diff = latest_commit.diff(json["embedded_commit"])

    # Print the changes (additions and deletions)
    print("Changes since the stored commit:")
    for change in diff:
        print(change)

    return True


if __name__ == "__main__":
    diff_hash()
