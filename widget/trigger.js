/**
 * Harbor — Proactive trigger snippet
 * Drop this on any client website to enable proactive chat greetings.
 * Fires after time-on-page, scroll depth, or exit intent.
 *
 * Usage: Include after the Chatwoot widget script.
 * Configure HARBOR_TRIGGER_CONFIG before the script loads.
 *
 * Example:
 *   window.HARBOR_TRIGGER_CONFIG = {
 *     timeDelay: 30,        // seconds before greeting (default: 30)
 *     scrollDepth: 50,      // % scrolled before greeting (default: 50)
 *     exitIntent: true,     // trigger on mouse leave (default: true)
 *     greeting: null,       // override greeting text (uses persona default if null)
 *   };
 */

(function () {
  const config = Object.assign(
    {
      timeDelay: 30,
      scrollDepth: 50,
      exitIntent: true,
      greeting: null,
    },
    window.HARBOR_TRIGGER_CONFIG || {}
  );

  let triggered = false;

  function openAndGreet() {
    if (triggered) return;
    triggered = true;

    // Open the Chatwoot widget
    if (window.$chatwoot) {
      window.$chatwoot.toggle("open");

      // Send trigger message if greeting is set
      // Chatwoot will relay this to Harbor which sends the persona greeting
      if (config.greeting) {
        // Short delay to let the widget open first
        setTimeout(() => {
          window.$chatwoot.setCustomAttributes({
            harbor_trigger: "proactive",
          });
        }, 500);
      }
    }
  }

  // Trigger 1: Time on page
  if (config.timeDelay > 0) {
    setTimeout(openAndGreet, config.timeDelay * 1000);
  }

  // Trigger 2: Scroll depth
  if (config.scrollDepth > 0) {
    window.addEventListener("scroll", function onScroll() {
      const scrolled =
        (window.scrollY /
          (document.documentElement.scrollHeight - window.innerHeight)) *
        100;
      if (scrolled >= config.scrollDepth) {
        window.removeEventListener("scroll", onScroll);
        openAndGreet();
      }
    });
  }

  // Trigger 3: Exit intent (desktop only)
  if (config.exitIntent) {
    document.addEventListener("mouseleave", function onMouseLeave(e) {
      if (e.clientY <= 0) {
        document.removeEventListener("mouseleave", onMouseLeave);
        openAndGreet();
      }
    });
  }
})();
