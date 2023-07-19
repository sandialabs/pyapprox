import numpy as np
import os


def convert_table_element_to_string(
        item, bold=False, underline=False, float_format="{0:.1e}"):
    if type(item) == int or type(item) == np.int64:
        latex_string = '%d' % item
    elif type(item) == float or type(item) == np.float64:
        latex_string = float_format.format(item)
    else:
        raise Exception
    if bold:
        latex_string = '\\bf{%s}' % latex_string
    if underline:
        latex_string = '\\underline{%s}' % latex_string
    return "& %s " % latex_string


def convert_to_latex_table(table_data, latex_filename, column_labels,
                           row_labels, caption, output_dir=None,
                           bold_entries=None, underline_entries=None,
                           width=None, corner_labels=None,
                           float_format="{0:.1e}"):
    """
    TODO: Currently assumes if have multiple rows that there are column
    labels for each row. Remove this assumption and check for either
    one row of column labels or column labels for each row
    """
    assert table_data.ndim == 2

    if width is None:
        width = '\\textwidth'

    if bold_entries is None:  # matrix the size of table data true_or_false
        bold_entries = np.empty_like(table_data)
        bold_entries[:, :] = False
    else:
        assert bold_entries.shape == table_data.shape

    if underline_entries is None:  # matrix the size of table data true_or_false
        underline_entries = np.empty_like(table_data)
        underline_entries[:, :] = False
    else:
        assert underline_entries.shape == table_data.shape

    table_spec = '|'
    for i in range(table_data.shape[1]+1):
        table_spec += 'c|'

        assert len(column_labels) == table_data.shape[1]*table_data.shape[0]
    assert len(row_labels) == table_data.shape[0]

    table_string = r'\hline '
    if corner_labels is not None:
        if len(corner_labels) == 2:
            table_string += r' \diagbox{%s}{%s} &' % (
                corner_labels[0], corner_labels[1])
        else:
            table_string += r' %s &' % (corner_labels[0])
    else:
        table_string += '&'

    def _print_column_labels(column_labels, table_string):
        for i in range(1, table_data.shape[1]):
            table_string += '%s &' % column_labels[i-1]
        table_string += r'%s \\' % column_labels[-1] + ' \n'
        return table_string
    for i in range(table_data.shape[0]):
        if i > 0:
            table_string += r'\hline &'
        table_string = _print_column_labels(
            column_labels[i*table_data.shape[1]:(i+1)*table_data.shape[1]],
            table_string)
        table_string += r'\hline '
        table_string += '%s ' % row_labels[i]
        print(row_labels[i])
        for j in range(table_data.shape[1]):
            table_string += convert_table_element_to_string(
                table_data[i, j], bold_entries[i, j],
                underline_entries[i, j], float_format)
        table_string += r"\\ "+"\n"
    table_string += r'\hline'
    print(table_string)
    file_string = r"""
\documentclass{standalone}
\usepackage{amsfonts,amsmath,amssymb,caption}
\usepackage{diagbox}
\usepackage[margin=1in]{geometry}
\captionsetup{justification=centering}
\begin{document}
\tiny
\minipage{%s}
\centering
\begin{tabular}[htb]{%s}
%s
\end{tabular}
""" % (width, table_spec, table_string)
    if caption is not None:
        file_string += r"\captionof*{table}{%s}" % (caption)
    file_string += r"""
        \endminipage
        \end{document}"""

    if output_dir is None:
        import tempfile
        output_dir = tempfile.mkdtemp()
    elif not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cur_dir = os.path.abspath(os.path.curdir)
    os.chdir(output_dir)

    with open(latex_filename, 'w') as f:
        f.write(file_string)

    import subprocess
    tex_out = subprocess.check_output(
        ['pdflatex', '--interaction=nonstopmode', latex_filename],
        stderr=subprocess.STDOUT)

    os.chdir(cur_dir)


def get_min_entry_per_row(table_data):
    assert table_data.ndim == 2
    min_entries = np.empty(table_data.shape)
    min_entries[:, :] = False
    I = np.argmin(table_data, axis=1)
    min_entries[np.arange(table_data.shape[0]), I] = True
    return min_entries


def test_convert_to_latex_table():
    table_data = np.arange(6.).reshape(2, 3)
    column_labels = ['', 'col1', 'col2', 'col3']
    row_labels = ['row1', 'row2']
    bold_entries = get_min_entry_per_row(table_data)
    convert_to_latex_table(table_data, 'table.tex', column_labels, row_labels,
                           'Table caption', output_dir='texdir',
                           bold_entries=bold_entries)

