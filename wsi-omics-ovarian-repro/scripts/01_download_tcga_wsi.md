
# Downloading TCGA ovarian diagnostic FFPE WSIs

1) GDC Data Portal: https://portal.gdc.cancer.gov/
2) Filters:
   - Project: TCGA-OV
   - Data Category: Biospecimen
   - Data Type: Tissue slide image
   - Experimental Strategy: Diagnostic Slide (filenames include DX#)
3) Add files to Cart -> Download manifest.
4) Install the GDC data transfer tool (gdc-client).
5) Run:
```bash
gdc-client download -m gdc_manifest.txt -d data/WSI/TCGA_OV
```
Helpful references:
- FFPE vs frozen and DX filtering notes: Andrew Janowczyk guide.
- ISB-CGC doc for TCGA pathology images in open GCS buckets.
