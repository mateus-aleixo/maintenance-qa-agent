"""Retrieval at millions of passages: the MS MARCO scale benchmark.

The serving system brute-forces everything because its corpus is 1,752 chunks. This
package measures what that design, and the indexes that would replace it, cost at
100 thousand, 1 million and 8.8 million passages. See scale/README.md.
"""
