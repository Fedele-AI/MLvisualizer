// @license magnet:?xt=urn:btih:1f739d935676111cfff4b4693e3816e664797050&dn=gpl-3.0.txt GPL-3.0-or-Later
(function(){
    // Initialize KaTeX auto-render if available. This small file is licensed under GPL-3.0-or-later.
    function runRender() {
        try {
            if (typeof renderMathInElement === 'function') {
                renderMathInElement(document.body, {delimiters: [{left: '$$', right: '$$', display: true}, {left: '$', right: '$', display: false}]});
            }
        } catch (e) {
            // ignore errors
        }
    }
    // Try immediately; also on window load as a fallback.
    runRender();
    if (typeof window !== 'undefined') {
        window.addEventListener('load', runRender, {once:true});
    }
})();
// @license-end
