// Collect Twitch VOD links from the current page DOM.
//
// How to use:
//   1. Open the page that lists the videos (e.g. a channel's "Videos" tab),
//      or a page with "preview card" links like:
//        <a data-a-target="preview-card-image-link" href="/videos/2843228444">
//   2. Open DevTools (F12) -> Console.
//   3. Paste this whole file, press Enter.
//   4. The list is printed and copied to the clipboard (one URL per line).
//
// Then feed the ids to TwitchDownloaderCLI, e.g.:
//   TwitchDownloaderCLI chatdownload --id 2843228444 -o chat.json
//   TwitchDownloaderCLI chatdownload --id 2843228444 -o chat.json --collision overwrite

(async () => {
  // Scroll the page to trigger lazy loading of more cards.
  // Set SCROLL_PASSES = 0 to disable scrolling and only read what is already loaded.
  const SCROLL_PASSES = 30;
  const SCROLL_DELAY_MS = 700;

  const collect = () => {
    const ids = new Set();

    // Any anchor pointing at a VOD, e.g.
    //   <a data-a-target="preview-card-image-link" href="/videos/2843228444">
    const links = document.querySelectorAll('a[href*="/videos/"]');

    for (const a of links) {
      const href = a.getAttribute("href") || "";
      const m = href.match(/\/videos\/(\d+)/);
      if (m) ids.add(m[1]);
    }
    return ids;
  };

  const all = new Set();

  for (let i = 0; i <= SCROLL_PASSES; i++) {
    collect().forEach((id) => all.add(id));

    if (i < SCROLL_PASSES) {
      window.scrollTo(0, document.body.scrollHeight);
      await new Promise((r) => setTimeout(r, SCROLL_DELAY_MS));
    }
  }

  const urls = [...all].map((id) => `https://www.twitch.tv/videos/${id}`);
  const text = urls.join("\n");

  console.log(`Found ${urls.length} video link(s):`);
  console.table(urls);

  // Copy to clipboard (falls back to the DevTools `copy()` helper).
  try {
    await navigator.clipboard.writeText(text);
    console.log("Copied to clipboard.");
  } catch {
    try {
      copy(text); // eslint-disable-line no-undef
      console.log("Copied to clipboard via copy().");
    } catch {
      console.log("Clipboard unavailable. Plain list:\n" + text);
    }
  }

  return urls;
})();
