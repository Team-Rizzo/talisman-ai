# Mechanism profiles

`profile.example.json` shows the SHAPE of a profile body. Every value in it is an inert
placeholder, deliberately: real settings are not kept in this repository.

Operating values live in the published profile, which is served to authenticated
validators only. Keep them there.

`version`, `publish_block` and `activation_block` are filled in by the publish command.

Nothing here is read at runtime. Validators only ever use what the API serves.
