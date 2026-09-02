# Data collection

GitHub Pages can host the rating page, but it cannot receive or store submitted
ratings by itself. Use a small external endpoint to write responses into a
shared table.

Recommended setup:

1. Create a Google Sheet for the study responses.
2. Open Extensions > Apps Script.
3. Paste the script from `collector/google-apps-script.js`.
4. Save the script.
5. Deploy it as a Web app.
6. For later script edits, open Deploy > Manage deployments, edit the Web app,
   choose Version > New version, then Deploy. Saving code alone does not update
   the `/exec` URL.
7. Set access to Anyone.
8. Put the deployed Web app URL into `config.js` as `submitEndpoint`.

After `submitEndpoint` is configured, the public page shows a Submit Online
button. Participants can still export CSV as a backup.

## Multiple study versions

The site can host multiple experiment versions. Each version can use the default
social robot criteria, or define its own criteria in `experiments.json`. Use URL
parameters to assign participants to a version:

- Complete version: `?experiment=complete`
- Single-LLM version: `?experiment=single-llm`
- Multi-round demo: `?experiment=multi-round-demo`

The participant sees only the public study label, such as `Sophia Study A` or
`Sophia Study F`. The exported and submitted data include `condition`,
`experiment_id`, `stimulus_set`, `order_seed`, and `dialogue_id` for analysis.

When new columns are added to `collector/google-apps-script.js`, paste the
updated script into Apps Script and redeploy the Web app so Google Sheets stores
the new fields.

The extra multi-study fields are appended after `completed_at`, so in Google
Sheets they appear near the right side of the response table. Existing rows from
older submissions will not be backfilled automatically; new submissions after
redeployment will include those values.
