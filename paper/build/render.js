
const { chromium } = require('playwright-core');
(async () => {
  const b = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome', args: ['--no-sandbox'] });
  const p = await b.newPage();
  await p.goto('file://' + process.argv[2]);
  await p.addScriptTag({ path: process.argv[4] + '/katex.min.js' });
  await p.addScriptTag({ path: process.argv[4] + '/contrib/auto-render.min.js' });
  await p.evaluate(() => renderMathInElement(document.body, { delimiters: [{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}], throwOnError: false }));
  await p.evaluate(() => document.fonts.ready);
  await p.waitForTimeout(800);
  await p.pdf({ path: process.argv[3], format: 'Letter', printBackground: true, preferCSSPageSize: true,
    displayHeaderFooter: true, headerTemplate: '<span></span>',
    footerTemplate: '<div style="width:100%;font-size:8px;font-family:Liberation Serif,serif;text-align:center;color:#333"><span class="pageNumber"></span></div>',
    margin: { top: '0.62in', bottom: '0.7in', left: '0.62in', right: '0.62in' } });
  const errs = await p.evaluate(() => document.querySelectorAll('.katex-error').length);
  console.log('katex errors:', errs);
  await b.close();
})();
