'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
test('plot themes show Plotly-based swatches, preserve legacy choices and update the selected value',()=>{
    const source=fs.readFileSync(require.resolve('../public/app.js'),'utf8');
    const context=vm.createContext({fileManagerEscape:s=>String(s).replaceAll('&','&amp;').replaceAll('"','&quot;').replaceAll('<','&lt;')});
    vm.runInContext(fs.readFileSync(require.resolve('../public/color-themes.js'),'utf8')+'\n'+source.slice(source.indexOf('const PLOT_THEME_CHOICES'),source.indexOf('/**\n * Open plot configuration modal'))+'\nthis.makePicker=plotColorPicker;this.select=selectPlotColor;',context);
    const html=context.makePicker('p1','Viridis');
    assert.match(html,/linear-gradient\(to right, #440154/);
    assert.equal((html.match(/data-color-theme=/g)||[]).length,7);
    assert.doesNotMatch(html,/data-color-theme="Jet"/);
    assert.match(context.makePicker('p1','Jet'),/data-color-theme="Jet"/);
    const input={value:''},current={textContent:''},swatch={style:{}},summary={focus(){this.focused=true;}};
    const picker={open:true,querySelector(selector){return {'input':input,'.plot-color-current':current,'summary .plot-color-swatch':swatch,'summary':summary}[selector];}};
    context.select({dataset:{colorTheme:'RdBu'},closest:()=>picker});
    assert.equal(input.value,'RdBu');assert.equal(current.textContent,'RdBu');assert.match(swatch.style.background,/linear-gradient/);assert.equal(picker.open,false);assert.ok(summary.focused);
    const page=fs.readFileSync(require.resolve('../public/index.html'),'utf8');
    assert.doesNotMatch(page,/data-plot-type="coarsening_mask"/);
    assert.ok(page.indexOf('color-themes.js')<page.indexOf('src="app.js"'));
});
