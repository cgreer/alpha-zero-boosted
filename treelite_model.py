from uuid import uuid4

from treelite import (
    Model as TreeliteModel,
    Annotator as TreeliteAnnotator,
    DMatrix as TreeliteDMatrix,
)

import settings


def build_annotation_data(
    model,
    annotation_samples,
    output_path,
    nthread=1,
):
    # sample_data_path in libsvm format
    samples_matrix = TreeliteDMatrix(annotation_samples)
    ann = TreeliteAnnotator()
    ann.annotate_branch(
        model=model,
        dmat=samples_matrix,
        nthread=nthread,
        verbose=False,
    )
    ann.save(path=output_path)
    print("Saved branch annotations in", output_path)
    return output_path


def build_treelite_model(
    gbdt_model_path,
    model_format="lightgbm",
    annotation_samples=None,
):
    # Load the model tree
    print("Loading lightgbm model")
    treelite_model = TreeliteModel.load(
        gbdt_model_path,
        model_format=model_format,
    )

    # Build up branch expectations from data samples
    print("Building branch expectations")
    annotation_results_path = None
    if annotation_samples is not None:
        tmp_annotation_info = f"{settings.TMP_DIRECTORY}/treelite_ann-{str(uuid4())}.info"
        # XXX: :nthread doesn't seem to do anything and only ever utilizes 1
        # core?
        annotation_results_path = build_annotation_data(
            treelite_model,
            annotation_samples,
            tmp_annotation_info,
            nthread=12, # XXX: pass through
        )

    # Compile model to C/C++
    print("Compiling Tree")
    params = dict(
        parallel_comp=settings.NUM_THREADS,
        # quantize=1, # Supposed to speed up predictions. Didn't when I tried it.
    )
    if annotation_results_path is not None:
        params["annotate_in"] = annotation_results_path

    # Save DLL
    treelite_model_path = f"{settings.TMP_DIRECTORY}/model-{str(uuid4())}.dylib"
    treelite_model.export_lib(
        toolchain=settings.TOOL_CHAIN, # clang for MacOS, gcc for unix?
        libpath=treelite_model_path,
        verbose=False,
        params=params,
    )
    print(f"Trained a treelite model: {treelite_model_path}")
    return treelite_model_path
