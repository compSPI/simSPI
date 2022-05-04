<!--
Thank you for opening this pull request (PR)!
-->

## Checklist

Verify that your PR checks all the following items.

- [ ] The pull request (PR) has a clear and explanatory title.
- [ ] The description of this PR links to relevant [GitHub issues](https://github.com/compSPI/simSPI/issues).
- [ ] Unit tests have been added in the [tests folder](https://github.com/compSPI/simSPI/tree/master/tests):
  - [ ] in the `test_*.py files` corresponding the files modified by this PR,
  - [ ] for each function added by this PR.
- [ ] The code of this PR is properly documented, with [docstrings following simSPI conventions](https://github.com/compspi/compspi/blob/master/docs/contributing.rst#the-anatomy-of-a-docstring).
- [ ] The PR passes the DeepSource GitHub Actions workflow (refer to comment below).
- [ ] The PR passes Test and Lint GitHub Actions workflows. (refer to comment below)

If some items are not checked, mark your PR as draft (Look for "Still in progress? Convert to Draft" on your PR) . Only mark the PR as "Ready for review" if all the items above are checked.

If you do not know how to address some items, reach out to a maintainer by requesting reviewers.

If some items cannot be addressed, explain the reason in the Description of your PR, and mark the PR ready for review

<!-- Checking that the PR passes the DeepSource workflow.

Check that the GitHub Action "DeepSource: Python" has passed. If it fails, click on "Details" to investigate and fix the issues.
-->

<!-- Checking that the PR passes the test workflow, i.e. passes the tests added in the tests folder.
First, run the tests related to your changes. For example, if you changed something in simSPI/crd.py:
$ pytest tests/test_crd.py

and then run the tests of the whole codebase to check that your feature is not breaking any of them:
$ pytest tests/

This way, further modifications on the code base are guaranteed to be consistent with the desired behavior. Merging your PR should not break any existing test.

Lastly, check that the tests run by GitHub Actions have passed, by scrolling down on your GitHub PR and verifying that the Actions such as "Test / build (ubuntu-18.04, 3.7, tests) (pull_request)" passed. If they fail, click on "Details" to investigate and fix the errors.
-->


<!-- Checking that the PR passes the lint
Install dependencies for developers
$ pip3 install -r dev-requirements.txt

Then run the following commands:
$ black . --check
$ isort --profile black --check .
$ flake8 simSPI tests

If some of these commands fail, you can either:
- fix the issues manually
- or use the pre-commit package as follows:

$ pre-commmit install
and then make a modification to one of your file to create a commit: the commit will clean your code automatically thanks to pre-commit. If pre-commit fails, remove it:
$ pre-commit uninstall

Lastly, check that the lint run by GitHub Actions has passed, by scrolling down on your GitHub PR and verifying that the Action such as "Lint / build (pull_request)" passed. If it failed, click on "Details" to investigate and fix the errors.
-->

## Description

<!-- Include a description of your pull request. If relevant, feel free to use this space to talk about time and space complexity as well scalability of your code-->

## Issue

<!-- Tell us which issue does this PR fix . Why this feature implementation/fix is important in practice ?-->

## Additional context

<!-- Add any extra information -->
