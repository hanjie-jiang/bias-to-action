window.MathJax = {
  options: {
    // Only typeset elements with class="arithmatex"
    ignoreHtmlClass: ".*",            // ignore everything
    processHtmlClass: "arithmatex"    // except these
  },
  tex: {
    inlineMath: [["$", "$"], ["\\(", "\\)"]],
    displayMath: [["$$", "$$"], ["\\[", "\\]"]],
    processEscapes: true,
    tags: "ams"
  }
};

document$.subscribe(() => {
  MathJax.typesetPromise();
});
