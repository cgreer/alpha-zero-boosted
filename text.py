

def stitch_text_blocks(text_blocks, delim="  "):
    decomposed_blocks = []
    for text_block in text_blocks:
        decomposed_blocks.append(text_block.strip().split("\n"))

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
