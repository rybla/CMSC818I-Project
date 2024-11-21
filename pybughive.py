from dataclasses import dataclass
import json
import subprocess
import os
from git import Repo
from diff_parser import Diff
from unidiff import PatchSet, PatchedFile

from utilities import debug_log

repository_path = "/users/henry/Documents/CMSC818I-Project/"

pybughive_dataset_filename = (
    f"{repository_path}pybughive/dataset/pybughive_current.json"
)


@dataclass
class ProjectIssue:
    username: str
    repository: str
    issue_index: int

    def dir(self):
        return f"/users/henry/Documents/cmsc818I/project/repositories/{self.repository}"


def get_project_info(project_issue: ProjectIssue) -> dict:
    with open(pybughive_dataset_filename, "r") as pybughive_dataset_file:
        dataset = json.load(pybughive_dataset_file)
        dataset = [
            pybughive_project
            for pybughive_project in dataset
            if pybughive_project["username"] == project_issue.username
            and pybughive_project["repository"] == project_issue.repository
        ]
        if len(dataset) == 0:
            raise Exception(
                f"no project {project_issue.username}/{project_issue.repository}"
            )
        project = dataset[0]
        return project


def get_project_issue_info(project_issue: ProjectIssue, project: dict) -> dict:
    issues = project["issues"]
    if not (project_issue.issue_index < len(issues)):
        raise Exception(f"issue index {project_issue.issue_index} out of bounds")
    issue = issues[project_issue.issue_index]
    return issue


@dataclass
class DiffBlock:
    patched_file: PatchedFile

    def line_ranges(self) -> tuple[tuple[int, int], tuple[int, int]]:
        for line in str(self.patched_file).split("\n"):
            if line.startswith("@@"):
                line = line[4:]
                i = line.find("@@")
                line = line[: i - 1]
                [a, b] = line.split(" ")
                a_ = a.split(",")
                a0 = int(a_[0])
                a1 = a0 + int(a_[1])
                b_ = b.split(",")
                b0 = int(b_[0])
                b1 = b0 + int(b_[1])
                return ((a0, a1), (b0, b1))

        raise Exception(f"invalid PatchedFile: {self.patched_file}")


def project_issue_diff_blocks(project_issue: ProjectIssue) -> list[DiffBlock]:
    debug_log(
        f"checking diff of {project_issue.username}/{project_issue.repository} before and after issue {project_issue.issue_index}"
    )
    project = get_project_info(project_issue)
    issue = get_project_issue_info(project_issue, project)
    buggy_commit_hash = issue["commits"][0]["hash"] + "^"
    fixed_commit_hash = issue["commits"][-1]["hash"]

    repo = Repo(".")
    repo.git.checkout(buggy_commit_hash)

    diff_blocks: list[DiffBlock] = []
    ps = PatchSet(repo.git.diff(fixed_commit_hash))
    pfs = ps.modified_files
    for pf in pfs:
        diff_blocks.append(DiffBlock(pf))

    return diff_blocks


def checkout_project_at_issue(project_issue: ProjectIssue):
    """
    Clones the project repository if necessary. Then, change directory to local
    clone of project repository, and checkout the commit corresponding to
    _before_ the bug was fixed.
    """
    debug_log(
        f"checking out project {project_issue.username}/{project_issue.repository} at issue {project_issue.issue_index}"
    )
    project = get_project_info(project_issue)
    issue = get_project_issue_info(project_issue, project)
    buggy_commit = issue["commits"][0]
    # fixed_commit = issue["commits"][-1]

    dir = project_issue.dir()
    if not os.path.exists(dir):
        debug_log(f"cloning project into {dir}")
        subprocess.run(
            f"git clone git@github.com:{project_issue.username}/{project_issue.repository}.git {dir}",
            shell=True,
        )
    else:
        debug_log(f"directory {dir} already exists")

    debug_log(f"changing directory to {dir}")
    os.chdir(dir)
    debug_log(f"checking out buggy commit {buggy_commit['hash']}")
    subprocess.run(f"git checkout -q {buggy_commit['hash']} --force", shell=True)


if __name__ == "__main__":
    checkout_project_at_issue(
        ProjectIssue(username="psf", repository="black", issue_index=0)
    )
    pass
