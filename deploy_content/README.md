# Deploy Content Overlay

Put files here when you want the next remote deploy to include them without
customizing the deploy tool.

Paths are relative to the repository root. For example:

```text
deploy_content/configs/seedon/blue_dynamic_support_gait.json
```

will be deployed as:

```text
configs/seedon/blue_dynamic_support_gait.json
```

Common examples:

```text
deploy_content/configs/seedon/my_experiment.json
deploy_content/tools/debug_my_case.py
deploy_content/seedon_baseline/env.py
```

Then run:

```bat
scripts\seedon_remote_deploy_and_check.bat
```

The deployer prints the overlay file count:

```text
Deploy overlay: D:\Git\robotlearning\seedon_mujoco\deploy_content (N files)
```

Rules:

- Keep the same relative path you want on the remote release.
- Use this for explicit payloads that are not ready to commit.
- Do not put models, logs, or large generated artifacts here.
- This directory is ignored by git except for this README and `.gitkeep`.
