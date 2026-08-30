# Data collection

GitHub Pages can host the rating page, but it cannot receive or store submitted
ratings by itself. Use a small external endpoint to write responses into a
shared table.

Recommended setup:

1. Create a Google Sheet for the study responses.
2. Open Extensions > Apps Script.
3. Paste the script from `collector/google-apps-script.js`.
4. Deploy it as a Web app.
5. Set access to Anyone.
6. Put the deployed Web app URL into `config.js` as `submitEndpoint`.

After `submitEndpoint` is configured, the public page shows a Submit Online
button. Participants can still export CSV as a backup.

## Multiple study versions

The site can host multiple experiment versions with the same rating criteria.
Use URL parameters to assign participants to a version:

- Complete version: `?experiment=complete`
- Single-LLM version: `?experiment=single-llm`

The participant sees only the public study label, such as `Sophia Study A` or
`Sophia Study B`. The exported and submitted data include `condition`,
`experiment_id`, `stimulus_set`, `order_seed`, and `dialogue_id` for analysis.

When new columns are added to `collector/google-apps-script.js`, paste the
updated script into Apps Script and redeploy the Web app so Google Sheets stores
the new fields.
