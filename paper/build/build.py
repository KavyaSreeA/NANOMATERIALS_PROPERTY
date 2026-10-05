"""Build paper.html and paper.pdf from paper_template.html.

1. Replace [[key,key]] markers by numbered IEEE citations (numbered by first appearance).
2. Append the reference list.
3. Render the page with headless Chromium (KaTeX typesets the formulas) and print it to PDF.

Run:  python3 build.py
"""
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent

REFS = {
 "haastrup": 'S. Haastrup <i>et al.</i>, &ldquo;The Computational 2D Materials Database: high-throughput modeling and discovery of atomically thin crystals,&rdquo; <i>2D Mater.</i>, vol. 5, no. 4, p. 042002, 2018.',
 "gjerding": 'M. N. Gjerding <i>et al.</i>, &ldquo;Recent progress of the Computational 2D Materials Database (C2DB),&rdquo; <i>2D Mater.</i>, vol. 8, no. 4, p. 044002, 2021.',
 "choudhary20": 'K. Choudhary <i>et al.</i>, &ldquo;The joint automated repository for various integrated simulations (JARVIS) for data-driven materials design,&rdquo; <i>npj Comput. Mater.</i>, vol. 6, p. 173, 2020.',
 "choudhary18": 'K. Choudhary, G. Cheon, E. Reid, and F. Tavazza, &ldquo;Elastic properties of bulk and low-dimensional materials using van der Waals density functional,&rdquo; <i>Phys. Rev. B</i>, vol. 98, p. 014107, 2018.',
 "venturi": 'V. Venturi, H. Parks, Z. Ahmad, and V. Viswanathan, &ldquo;Machine learning enabled discovery of application dependent design principles for two-dimensional materials,&rdquo; arXiv:2003.13418, 2020.',
 "fronzi": 'M. Fronzi, M. Abu Ghazaleh, O. Isayev, D. A. Winkler, J. Shapter, and M. J. Ford, &ldquo;Impressive computational acceleration by using machine learning for 2-dimensional super-lubricant materials discovery,&rdquo; arXiv:1911.11559, 2020.',
 "malakar": 'P. Malakar, M. S. H. Thakur, S. M. Nahid, and M. M. Islam, &ldquo;Data-driven machine learning to predict mechanical properties of monolayer TMDs,&rdquo; manuscript (venue not stated in the copy reviewed).',
 "maurizi": 'M. Maurizi, C. Gao, and F. Berto, &ldquo;Predicting stress, strain and deformation fields in materials and structures with graph neural networks,&rdquo; <i>Sci. Rep.</i>, vol. 12, p. 21834, 2022.',
 "davidovikj": 'D. Davidovikj <i>et al.</i>, &ldquo;Young&rsquo;s modulus of 2D materials extracted from their nonlinear dynamic response,&rdquo; arXiv:1704.05433, 2017.',
 "elder": 'R. M. Elder, M. R. Neupane, and T. L. Chantawansri, &ldquo;Mechanical properties of homogeneous and heterogeneous layered 2D materials,&rdquo; in <i>Proc. SISPAD</i> (IEEE).',
 "zhang": 'R. Zhang and R. Cheung, &ldquo;Mechanical properties and applications of two-dimensional materials,&rdquo; book chapter, InTech, 2016.',
 "ward16": 'L. Ward, A. Agrawal, A. Choudhary, and C. Wolverton, &ldquo;A general-purpose machine learning framework for predicting properties of inorganic materials,&rdquo; <i>npj Comput. Mater.</i>, vol. 2, p. 16028, 2016.',
 "ward18": 'L. Ward <i>et al.</i>, &ldquo;Matminer: An open source toolkit for materials data mining,&rdquo; <i>Comput. Mater. Sci.</i>, vol. 152, pp. 60&ndash;69, 2018.',
 "ong": 'S. P. Ong <i>et al.</i>, &ldquo;Python Materials Genomics (pymatgen): A robust, open-source Python library for materials analysis,&rdquo; <i>Comput. Mater. Sci.</i>, vol. 68, pp. 314&ndash;319, 2013.',
 "ke": 'G. Ke <i>et al.</i>, &ldquo;LightGBM: A highly efficient gradient boosting decision tree,&rdquo; in <i>Proc. NeurIPS</i>, vol. 30, 2017.',
 "pedregosa": 'F. Pedregosa <i>et al.</i>, &ldquo;Scikit-learn: Machine learning in Python,&rdquo; <i>J. Mach. Learn. Res.</i>, vol. 12, pp. 2825&ndash;2830, 2011.',
 "breiman": 'L. Breiman, &ldquo;Random forests,&rdquo; <i>Mach. Learn.</i>, vol. 45, no. 1, pp. 5&ndash;32, 2001.',
 "hoerl": 'A. E. Hoerl and R. W. Kennard, &ldquo;Ridge regression: Biased estimation for nonorthogonal problems,&rdquo; <i>Technometrics</i>, vol. 12, no. 1, pp. 55&ndash;67, 1970.',
 "meredig": 'B. Meredig <i>et al.</i>, &ldquo;Can machine learning identify the next high-temperature superconductor? Examining extrapolation performance for materials discovery,&rdquo; <i>Mol. Syst. Des. Eng.</i>, vol. 3, pp. 819&ndash;825, 2018.',
 "roberts": 'D. R. Roberts <i>et al.</i>, &ldquo;Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure,&rdquo; <i>Ecography</i>, vol. 40, pp. 913&ndash;929, 2017.',
 "mortensen": 'J. J. Mortensen, L. B. Hansen, and K. W. Jacobsen, &ldquo;Real-space grid implementation of the projector augmented wave method,&rdquo; <i>Phys. Rev. B</i>, vol. 71, p. 035109, 2005.',
 "pbe": 'J. P. Perdew, K. Burke, and M. Ernzerhof, &ldquo;Generalized gradient approximation made simple,&rdquo; <i>Phys. Rev. Lett.</i>, vol. 77, pp. 3865&ndash;3868, 1996.',
 "kresse": 'G. Kresse and J. Furthm&uuml;ller, &ldquo;Efficient iterative schemes for <i>ab initio</i> total-energy calculations using a plane-wave basis set,&rdquo; <i>Phys. Rev. B</i>, vol. 54, pp. 11169&ndash;11186, 1996.',
 "klimes": 'J. Klime&scaron;, D. R. Bowler, and A. Michaelides, &ldquo;Van der Waals density functionals applied to solids,&rdquo; <i>Phys. Rev. B</i>, vol. 83, p. 195131, 2011.',
 "batatia": 'I. Batatia <i>et al.</i>, &ldquo;A foundation model for atomistic materials chemistry,&rdquo; arXiv:2401.00096, 2023.',
 "deng": 'B. Deng <i>et al.</i>, &ldquo;CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling,&rdquo; <i>Nat. Mach. Intell.</i>, vol. 5, pp. 1031&ndash;1041, 2023.',
 "larsen": 'A. H. Larsen <i>et al.</i>, &ldquo;The atomic simulation environment&mdash;a Python library for working with atoms,&rdquo; <i>J. Phys.: Condens. Matter</i>, vol. 29, p. 273002, 2017.',
 "grimme": 'S. Grimme, J. Antony, S. Ehrlich, and H. Krieg, &ldquo;A consistent and accurate <i>ab initio</i> parametrization of density functional dispersion correction (DFT-D) for the 94 elements H&ndash;Pu,&rdquo; <i>J. Chem. Phys.</i>, vol. 132, p. 154104, 2010.',
 "spearman": 'C. Spearman, &ldquo;The proof and measurement of association between two things,&rdquo; <i>Am. J. Psychol.</i>, vol. 15, no. 1, pp. 72&ndash;101, 1904.',
 "efron": 'B. Efron, &ldquo;Bootstrap methods: Another look at the jackknife,&rdquo; <i>Ann. Statist.</i>, vol. 7, no. 1, pp. 1&ndash;26, 1979.',
}

html = (HERE / "paper_template.html").read_text(encoding="utf-8")
order = []


def cite(m):
    keys = [k.strip() for k in m.group(1).split(",")]
    nums = []
    for k in keys:
        if k not in REFS:
            raise KeyError(k)
        if k not in order:
            order.append(k)
        nums.append(order.index(k) + 1)
    return "[" + "], [".join(str(n) for n in nums) + "]"


body = re.sub(r"\[\[([a-z0-9, ]+)\]\]", lambda m: cite(m) if m.group(1) != "REFLIST" else m.group(0), html)
reflist = "\n".join(f"<div>[{i + 1}] {REFS[k]}</div>" for i, k in enumerate(order))
body = body.replace("[[REFLIST]]", reflist)
unused = [k for k in REFS if k not in order]
print("cited", len(order), "unused refs:", unused)
body = body.replace('href="node_modules/katex/dist/katex.min.css"', 'href="build/node_modules/katex/dist/katex.min.css"')
(OUT / "paper.html").write_text(body, encoding="utf-8")

# render + print with Chromium
js = r"""
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
"""
(HERE / "render.js").write_text(js)
subprocess.run(["node", "render.js", str(OUT / "paper.html"), str(OUT / "paper.pdf"),
                str(HERE / "node_modules/katex/dist")], cwd=HERE, check=True)
print("wrote", OUT / "paper.pdf")
