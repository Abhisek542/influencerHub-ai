"""Empties the local demo database container and seeds it again. Only touches influencerhub-demo-pg."""
import subprocess

import seed_demo

SQL = 'TRUNCATE campaign_application, campaign, userprofile, users RESTART IDENTITY CASCADE'


def reset() -> None:
    subprocess.run(
        [
            "docker",
            "exec",
            "influencerhub-demo-pg",
            "sh",
            "-c",
            f'psql -v ON_ERROR_STOP=1 -q -U "$POSTGRES_USER" -d influencerhub_demo -c "{SQL}"',
        ],
        check=True,
    )
    seed_demo.main()


if __name__ == "__main__":
    reset()
