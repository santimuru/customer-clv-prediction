"""Visual QA: click each top-nav section, scroll-tile screenshots."""
import time, os
from playwright.sync_api import sync_playwright

OUT = "C:/Users/biconsulting/portfolio/customer-clv-prediction/_qa"
os.makedirs(OUT, exist_ok=True)
for f in os.listdir(OUT):
    os.remove(os.path.join(OUT, f))

URL = "http://localhost:8602/"
NAV = ["Ledger", "Segments", "Forecast", "Simulator"]
VH = 1000

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1500, "height": VH})
    pg.goto(URL, wait_until="networkidle", timeout=60000)
    time.sleep(4)
    for si, label in enumerate(NAV):
        clicked = pg.evaluate("""(lbl) => {
          const labels=[...document.querySelectorAll('div[data-testid="stRadio"] label')];
          const el=labels.find(e=>e.innerText.trim().toLowerCase()===lbl.toLowerCase());
          if(el){el.click(); return true;} return false; }""", label)
        time.sleep(3.2)
        cont = pg.query_selector('[data-testid="stMainBlockContainer"]')
        total = pg.evaluate("(e)=>e.scrollHeight", cont) if cont else VH
        n = max(1, (total // VH) + 1)
        for i in range(n):
            pg.evaluate(f"()=>document.querySelector('[data-testid=stMain]').scrollTo(0,{i*VH})")
            time.sleep(0.9)
            pg.screenshot(path=f"{OUT}/{si:02d}_{label}_{i:02d}.png")
        print(f"{label}: clicked={clicked} {n} tiles ({total}px)")
    # also exercise the simulator submit
    pg.evaluate("""()=>{const e=[...document.querySelectorAll('div[data-testid=stRadio] label')].find(x=>x.innerText.trim().toLowerCase()==='simulator');e&&e.click();}""")
    time.sleep(2.5)
    ok = pg.evaluate("""()=>{const b=[...document.querySelectorAll('button')].find(x=>x.innerText.trim().toLowerCase().includes('predict clv'));if(b){b.click();return true}return false}""")
    time.sleep(5)
    pg.evaluate("()=>document.querySelector('[data-testid=stMain]').scrollTo(0,500)")
    time.sleep(1.2)
    pg.screenshot(path=f"{OUT}/99_sim_result.png")
    err = pg.evaluate("""()=>[...document.querySelectorAll('[data-testid=stException],.stException')].map(e=>e.innerText).slice(0,3)""")
    print("submit:", ok, "errors:", err)
    b.close()
