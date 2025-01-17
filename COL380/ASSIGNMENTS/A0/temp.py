import os
from pdf2image import convert_from_path
import subprocess

# LaTeX document content
latex_document = r"""
\documentclass{article}
\usepackage{booktabs}
\begin{document}

\section*{Performance and Profiling Table}
The table below displays performance and profiling data:

\begin{tabular}{lrr}
\hline
       & perf   & gprof  \\
\hline
multiply & n/a    & n/a    \\
read     & n/a    & n/a    \\
write    & n/a    & n/a    \\
main     & n/a    & n/a    \\
\hline
\end{tabular}

\end{document}
"""

# Write LaTeX to a file
with open("main.tex", "w") as file:
    file.write(latex_document)

# Compile the LaTeX file to PDF using pdflatex
subprocess.run(["pdflatex", "main.tex"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Convert the PDF to an image
pdf_path = "main.pdf"
images = convert_from_path(pdf_path)

# Save the first page of the PDF as an image
image_path = "main.png"
images[0].save(image_path, "PNG")

# Clean up auxiliary files
for ext in ["aux", "log", "pdf", "tex"]:
    os.remove(f"main.{ext}")

print(f"Image saved as {image_path}")
