# Publication Deduplicator

## Overview

The Deduplicator is a Streamlit-based application designed to process large collections of research article references and produce a clean, deduplicated, and export-ready list. The tool is particularly suited for academic reporting, annual reports, institutional submissions, and accreditation documents where duplicate references, inconsistent formatting, and incomplete metadata are common issues.

The application combines rule-based text parsing, hash-based comparison, DOI matching, and fuzzy string similarity to reliably identify and remove duplicate references while preserving essential metadata such as NAAS ratings, Impact Factors, and Crossref DOI links.

---

## Objectives

The primary objectives of this application are:

1. To accurately detect and remove duplicate research article references.
2. To support heterogeneous input formats commonly used in academic documents.
3. To preserve bibliometric indicators such as NAAS ratings and Impact Factors when present.
4. To generate clean, unnumbered outputs suitable for formal documentation.
5. To provide transparent verification of detected article counts and duplicates.

---

## Supported Input Formats

The application supports the following reference formats:

### 1. Numbered References

References listed with explicit numbering at the beginning of each line.

Example:

```

1. Author A. (2021). Title. Journal Name, Volume(Issue), Pages. DOI
2. Author B. (2020). Title. Journal Name, Volume(Issue), Pages. DOI

```

### 2. Paragraph-Separated References

References separated by blank lines, often seen in copied bibliographies or DOCX files.

### 3. DOCX Uploads

Microsoft Word documents containing references in either numbered or paragraph format.

---

## Processing Pipeline

The processing workflow consists of the following sequential stages.

---

## 1. Format Detection

Before parsing, the input text is analyzed to determine its dominant structure.

The detection logic considers:

- The number of lines starting with a numeric pattern (`^\d+\.`)
- The number of paragraph breaks
- The presence of common author name patterns

Based on heuristic thresholds, the input is classified as:

- Numbered
- Paragraph-based
- Mixed

This classification determines the parsing strategy.

---

## 2. Article Parsing

### Numbered Parsing

If the input is detected as numbered and numbering is trusted, articles are extracted using the pattern:

```

(\d+).\s+(article_text)

```

This allows:

- Detection of missing numbers
- Verification of expected vs detected article counts

### Paragraph Parsing

For non-numbered or mixed inputs:

- Text is split using blank lines
- Short or invalid blocks are discarded
- Blocks are validated using year and author-name heuristics

---

## 3. Text Normalization

Each article is normalized before comparison to minimize false mismatches caused by formatting differences.

Normalization steps include:

- Conversion to lowercase
- Removal of extra whitespace
- Removal of URLs and DOI strings
- Removal of NAAS-related annotations
- Removal of publication date suffixes

Let the original article string be `A`.
The normalized form `N(A)` is produced by applying the above transformations.

---

## 4. Hash-Based Duplicate Detection

For each normalized article `N(A)`, a hash is generated using MD5:

```

H(A) = MD5(N(A))

```

Articles with identical hashes are treated as exact duplicates.

This provides:

- Fast initial grouping
- Deterministic duplicate detection

---

## 5. DOI-Based Matching

If a DOI is present, it is treated as a strong identifier.

Let `D(A)` be the extracted DOI for article `A`.

If:

```

D(A₁) = D(A₂)

```

then articles `A₁` and `A₂` are considered duplicates even if their text differs.

This step is particularly important when:

- Author names are formatted differently
- Journal names are abbreviated inconsistently

---

## 6. Fuzzy Similarity Matching

For articles not matched by hash or DOI, fuzzy similarity is applied using the SequenceMatcher ratio.

For two normalized articles `N(A₁)` and `N(A₂)`, similarity is computed as:

```

S(A₁, A₂) = SequenceMatcher(N(A₁), N(A₂)).ratio()

```

If:

```

S(A₁, A₂) ≥ τ

```

where `τ` is a user-defined threshold (default = 0.85), the articles are considered duplicates.

This step handles:

- Minor typographical differences
- Variations in punctuation
- Slight reordering of elements

---

## 7. Duplicate Group Resolution

For each group of duplicates:

- The earliest occurrence is retained
- All subsequent occurrences are marked as duplicates
- A mapping of retained vs removed indices is maintained

This ensures reproducibility and transparency.

---

## 8. Metadata Extraction

### DOI Extraction

DOIs are extracted using multiple regular expression patterns to account for:

- DOI URLs
- DOI prefixed strings
- Raw DOI identifiers

### NAAS and Impact Factor Extraction

NAAS ratings and Impact Factors are extracted using pattern matching such as:

```

NAAS 5.24
(NAAS rating 4.8)
Thomson Reuters IF: 3.2
(IF: 2.9)

```

The application does not compute or infer ratings; it only preserves what is explicitly present.

---

## 9. Crossref Integration

For articles with missing DOIs:

- The title and publication year are extracted
- A Crossref API query is issued
- Returned titles are validated using fuzzy similarity

If similarity exceeds the acceptance threshold, the DOI is attached and a resolvable DOI link is generated.

Rate limiting is enforced to comply with Crossref usage policies.

---

## 10. Output Generation

### TXT Output

- Clean, unnumbered references
- Includes NAAS/IF annotations if present
- Includes Crossref DOI links

### DOCX Output

- Identical content to TXT
- Suitable for annexures, reports, and submissions
- No forced numbering

### On-Screen Display

- Numbered for readability
- Interactive duplicate inspection
- Expandable duplicate groups

---

## Frequency Analysis

A frequency table is generated showing:

- Number of occurrences per unique article
- Presence of DOI
- Presence of ratings
- Duplicate status

This helps users understand redundancy patterns in their data.

---

## Assumptions and Limitations

- NAAS ratings and Impact Factors are not calculated automatically
- Journal Impact Factors are not fetched from external databases
- Fuzzy similarity thresholds may require tuning for highly heterogeneous datasets
- Auto-fetching DOIs depends on Crossref availability and rate limits

---

## Intended Use Cases

- Annual academic reports
- Faculty publication audits
- Accreditation submissions
- Institutional research summaries
- Grant documentation

---

## Technology Stack

- Python
- Streamlit
- Crossref REST API
- python-docx
- pandas

---

## Developers

- **Dr. Sandip Garai** - [Google Scholar](https://scholar.google.com/citations?user=Es-kJk4AAAAJ&hl=en)
- **Dr. Kanaka K K** - [Google Scholar](https://scholar.google.com/citations?user=0dQ7Sf8AAAAJ&hl=en&oi=ao)

📧 [Contact:](mailto:drgaraislab@gmail.com)

---

## License

This project is licensed under the MIT License - see the [LICENSE](https://mit-license.org/) file for details.

---

## Acknowledgments
Streamlit team for the excellent web framework

---

**Publication Deduplicator v1.0** | Developed for Bibliography Preparation




