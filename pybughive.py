from dataclasses import dataclass
import json
import subprocess
import os

from utilities import debug_log

pybughive_dataset_filename = "./pybughive/dataset/pybughive_current.json"


@dataclass
class ProjectIssue:
    username: str
    repository: str
    issue_index: int

    def get_directory(self):
        return f"./repositories/{self.repository}"


# def checkout_project_at_issue(username: str, repository: str, issue_index: int):
def checkout_project_at_issue(project_issue: ProjectIssue):
    debug_log(
        f"checking out project {project_issue.username}/{project_issue.repository} at issue {project_issue.issue_index}"
    )
    with open(pybughive_dataset_filename) as pybughive_dataset_file:
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

        username_ = project["username"]
        repository_ = project["repository"]
        issues = project["issues"]
        if not (project_issue.issue_index < len(issues)):
            raise Exception(f"issue index {project_issue.issue_index} out of bounds")
        issue = issues[project_issue.issue_index]
        buggy_commit = issue["commits"][0]
        fixed_commit = issue["commits"][-1]

        dir = project_issue.get_directory()
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
