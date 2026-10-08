import os
import json
import requests
import datetime

def get_data():
    base = "http://api.worldbank.org/v2/country/all/indicator/"
    inds = [
        "SP.DYN.LE00.IN",
        "NY.GDP.PCAP.CD",
        "SP.DYN.IMRT.IN",
        "SH.XPD.CHEX.GD.ZS",
        "EG.ELC.ACCS.ZS",
        "IT.NET.USER.ZS",
        "SP.URB.TOTL.IN.ZS",
        "SL.UEM.TOTL.ZS",
        "SP.POP.TOTL",
        "SH.H2O.BASW.ZS",
        "SP.DYN.TFRT.IN",
        "NY.GDP.MKTP.KD.ZG"
    ]
    meta = "http://api.worldbank.org/v2/country/all?format=json&per_page=20000"
    srcs = {}
    dt = datetime.datetime.now(datetime.timezone.utc).isoformat()
    raw = os.path.join("data", "raw")
    os.makedirs(raw, exist_ok=True)
    try:
        rm = requests.get(meta, timeout=30, verify=False)
        rm.raise_for_status()
        fm = os.path.join(raw, "meta.json")
        with open(fm, "w", encoding="utf-8") as f:
            f.write(rm.text)
        srcs["meta"] = {"url": meta, "licence": "CC BY 4.0", "date": dt}
        for ind in inds:
            url = f"{base}{ind}?format=json&per_page=20000"
            res = requests.get(url, timeout=30, verify=False)
            res.raise_for_status()
            d = res.json()
            if len(d) > 1 and len(d[1]) > 0:
                fi = os.path.join(raw, f"{ind}.json")
                with open(fi, "w", encoding="utf-8") as f:
                    f.write(res.text)
                srcs[ind] = {"url": url, "licence": "CC BY 4.0", "date": dt}
            else:
                print(f"Skipping {ind}")
        fs = os.path.join(raw, "sources.json")
        with open(fs, "w", encoding="utf-8") as f:
            json.dump(srcs, f, indent=4)
        print("Data downloaded")
    except Exception as e:
        print(f"API unreachable fallback to cached data: {e}")

if __name__ == "__main__":
    get_data()
