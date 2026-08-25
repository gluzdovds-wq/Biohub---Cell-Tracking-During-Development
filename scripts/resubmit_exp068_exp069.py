from __future__ import annotations

from dataclasses import dataclass

from kaggle.api.kaggle_api_extended import KaggleApi


COMPETITION = "biohub-cell-tracking-during-development"


@dataclass(frozen=True)
class Retry:
    kernel: str
    version: int
    description: str


RETRIES = (
    Retry(
        kernel="pawanmali/biohub-mcflow-v1",
        version=2,
        description="EXP-068R full-inference min-cost-flow v2 resubmit",
    ),
    Retry(
        kernel="flexonafft/biohub-harmonic-fusion",
        version=11,
        description="EXP-069R full-inference harmonic-fusion v11 resubmit",
    ),
)


def main() -> None:
    api = KaggleApi()
    api.authenticate()
    prior_descriptions = {
        submission.description
        for submission in api.competition_submissions(COMPETITION)
    }

    for retry in RETRIES:
        if retry.description in prior_descriptions:
            print(f"SKIP already registered: {retry.description}")
            continue
        response = api.competition_submit_code(
            file_name="submission.csv",
            message=retry.description,
            competition=COMPETITION,
            kernel=retry.kernel,
            kernel_version=retry.version,
        )
        print(
            f"SUBMITTED ref={response.ref} kernel={retry.kernel} "
            f"version={retry.version}"
        )


if __name__ == "__main__":
    main()
