
def stitch_text_blocks(text_blocks, delim="  "):
    '''
    XXX: handle colors by removing them from the row length adjustment calculations
    '''
    decomposed_blocks = []
    for i, text_block in enumerate(text_blocks):
        block_rows = text_block.split("\n")
        block_rows = [row.strip() for row in block_rows]
        block_rows = [row for row in block_rows if row]

        max_row_length = max(len(row) for row in block_rows)

        length_adjusted_block_rows = []
        for block_row in block_rows:
            while len(block_row) < max_row_length:
                block_row += " "
            length_adjusted_block_rows.append(block_row)

        decomposed_blocks.append(length_adjusted_block_rows)

    # Make all dblocks the same length
    max_length = max(len(x) for x in decomposed_blocks)
    for dblock in decomposed_blocks:
        while len(dblock) < max_length:
            dblock.append("")

    # Stitch
    stitched_blocks = ""
    for drows in zip(*decomposed_blocks):
        frow = delim.join(drows) + "\n"
        stitched_blocks += frow

    return stitched_blocks
