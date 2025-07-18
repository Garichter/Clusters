import cdsapi

c = cdsapi.Client()

c.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": "reanalysis",
        "variable": "total_precipitation",
        "year": "2025",
        "month": "06",
        "day": ["17"],
        "time": [f"{h:02d}:00" for h in range(24)],
        "format": "grib",
        "area": [-27, -57, -34, -48],  # N, W, S, E (aproximação do RS)
    },
    "precip_rs_jun172025.grib"
)