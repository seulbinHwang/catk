Canonical location for the reusable Waymo login session:

- `secrets/waymo/waymo_storage_state.json`

This JSON file is effectively an authenticated browser session for your Waymo account.
It may also contain Waymo challenge localStorage state (for example the
`datasetChallengeTermsAgreementAccepted` flag used by the Sim Agents upload form).
Anyone who can read it can usually submit on behalf of that account until the session expires
or is revoked.

If you keep the raw file in git, do it only in a private repo with tightly controlled access.
Safer options are `git-crypt`, `sops`, or keeping the raw JSON untracked and distributing it
through a secrets manager.

For server-side auto submission in this repo, you can also avoid copying the raw file entirely:
run with `waymo_submission.enabled=true`, then paste the local JSON into the terminal prompt when
rank 0 asks for it at startup.
