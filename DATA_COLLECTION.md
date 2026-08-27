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
