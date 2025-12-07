'''
Utility functions for querying Git metadata,
used for experiment tracking and reproducibility.
'''
import subprocess
import os
import logging

logger = logging.getLogger(__name__)


def get_git_metadata(prefix: str = 'git') -> dict:
    ''' Query Git metagdata: (branch, commit, username)

    Locally uses git commands, but prefers environment variables used in CI systems.
    This makes the function reliable in GitHub Actions where the checkout may be a
    detached HEAD; Actions set useful environment variables we prefer.

    Priority order:
      - For commit:     GIT_COMMIT,     GITHUB_SHA,                     `git rev-parse HEAD`.
      - For branch:     GIT_BRANCH,     GITHUB_HEAD_REF (PR source),    `git rev-parse --abbrev-ref HEAD`
      - For user:       GIT_USER,       GITHUB_ACTOR,                    git config user.name

    Returns
    -------
    metadata: dict
        Dict with keys '{prefix}.branch', '{prefix}.commit', '{prefix}.user'
        If any value cannot be determined, 'unknown' is used.
    '''
    commit = os.environ.get('GIT_COMMIT') or os.environ.get('GITHUB_SHA') or __git_command(['rev-parse', 'HEAD'])

    branch = os.environ.get('GIT_BRANCH') or os.environ.get('GITHUB_HEAD_REF') or __git_command(['rev-parse', '--abbrev-ref', 'HEAD'])

    user = os.environ.get('GIT_USER') or os.environ.get('GITHUB_ACTOR') or __git_command(['config', 'user.name'])

    values = {
        'branch': branch if branch else 'unknown',
        'commit': commit if commit else 'unknown',
        'user': user if user else 'unknown',
    }
    values = {f'{prefix}.{k}': v for k, v in values.items()} # Return with prefix for tagging
    logger.debug(f"Queried git metadata: {values}")
    return values


def __git_command(args):
    ''' Run a git command on terminal.
    
    Parameters
    ----------
    args: list
        List of git command arguments, e.g. ['rev-parse', 'HEAD']
    
    Returns
    -------
    output: str or None
        The command output as string if successful, else None
    '''
    try:
        completed = subprocess.run(['git'] + args, capture_output=True, text=True, check=False)
        if completed.returncode == 0:
            return completed.stdout.strip()
    except Exception:
        pass
    return None