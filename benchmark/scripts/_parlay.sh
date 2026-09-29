#!/usr/bin/env bash
# Published ParlayANN recipes shared by preparation and comparison wrappers.
# These are PA-specific build/query settings, not Orion cascade overrides.
load_parlay_recipe() {
  local DATASET="$1"
case "$DATASET" in
    sift)
        # `vamana/scripts/sift`:
        #   BUILD_ARGS="-R 64 -L 128 -alpha 1.15 -num_passes 2 -quantize_bits 8 -verbose"
        #   QUERY_ARGS="-quantize_bits 8 -verbose"
        #   TYPE_ARGS="-data_type float -dist_func Euclidian -file_type bin"
        BASE_FVECS="data/sift/sift_base.fvecs"
        QUERY_FVECS="data/sift/sift_query.fvecs"
        GT_IVECS="data/sift/sift_groundtruth.ivecs"
        PA_DIR_NAME="sift1m"
        DEFAULT_MAX_POINTS=""
        PA_BUILD_R=64
        PA_BUILD_L=128
        PA_BUILD_ALPHA=1.15
        PA_BUILD_PASSES=2
        PA_BUILD_QBITS=8
        PA_DIST_FUNC="Euclidian"
        PA_NORMALIZE_FLAG=""
        PA_QUERY_ARGS=(-quantize_bits 8 -verbose)
        DEFAULT_MAX_EXTRA=16
        ;;
    glove25)
        # `vamana/scripts/glove25`:
        #   BUILD_ARGS="-R 100 -L 200 -alpha 1 -num_passes 2 -quantize_bits 8 -verbose"
        #   QUERY_ARGS="-quantize_bits 16 -quantize_mode 1 -verbose -rerank_factor 2"
        #   TYPE_ARGS="-data_type float -dist_func mips -normalize -file_type bin"
        BASE_FVECS="data/glove25/glove-25-angular_base.fvecs"
        QUERY_FVECS="data/glove25/glove-25-angular_query.fvecs"
        GT_IVECS="data/glove25/glove-25-angular_groundtruth.ivecs"
        PA_DIR_NAME="glove25"
        DEFAULT_MAX_POINTS=""
        PA_BUILD_R=100
        PA_BUILD_L=200
        PA_BUILD_ALPHA=1
        PA_BUILD_PASSES=2
        PA_BUILD_QBITS=8
        PA_DIST_FUNC="mips"
        PA_NORMALIZE_FLAG="-normalize"
        PA_QUERY_ARGS=(-quantize_bits 16 -quantize_mode 1 -verbose -rerank_factor 2)
        DEFAULT_MAX_EXTRA=16
        ;;
    glove100)
        # `vamana/scripts/glove100`:
        #   BUILD_ARGS="-R 100 -L 200 -alpha 1 -num_passes 2 -quantize_bits 8 -verbose"
        #   QUERY_ARGS="-quantize_bits 16 -quantize_mode 1 -verbose -rerank_factor 2"
        #   TYPE_ARGS="-data_type float -dist_func mips -normalize -file_type bin"
        BASE_FVECS="data/glove100/glove-100-angular_base.fvecs"
        QUERY_FVECS="data/glove100/glove-100-angular_query.fvecs"
        GT_IVECS="data/glove100/glove-100-angular_groundtruth.ivecs"
        PA_DIR_NAME="glove100"
        DEFAULT_MAX_POINTS=""
        PA_BUILD_R=100
        PA_BUILD_L=200
        PA_BUILD_ALPHA=1
        PA_BUILD_PASSES=2
        PA_BUILD_QBITS=8
        PA_DIST_FUNC="mips"
        PA_NORMALIZE_FLAG="-normalize"
        PA_QUERY_ARGS=(-quantize_bits 16 -quantize_mode 1 -verbose -rerank_factor 2)
        DEFAULT_MAX_EXTRA=16
        ;;
    gist)
        # `vamana/scripts/gist`:
        #   BUILD_ARGS="-R 100 -L 200 -alpha 1.1 -num_passes 2 -quantize_bits 8 -verbose"
        #   QUERY_ARGS="-quantize_bits 16 -quantize_mode 3 -verbose -rerank_factor 2"
        #   TYPE_ARGS="-data_type float -dist_func Euclidian -file_type bin"
        BASE_FVECS="data/gist/gist_base.fvecs"
        QUERY_FVECS="data/gist/gist_query.fvecs"
        GT_IVECS="data/gist/gist_groundtruth.ivecs"
        PA_DIR_NAME="gist"
        DEFAULT_MAX_POINTS=""
        PA_BUILD_R=100
        PA_BUILD_L=200
        PA_BUILD_ALPHA=1.1
        PA_BUILD_PASSES=2
        PA_BUILD_QBITS=8
        PA_DIST_FUNC="Euclidian"
        PA_NORMALIZE_FLAG=""
        PA_QUERY_ARGS=(-quantize_bits 16 -quantize_mode 3 -verbose -rerank_factor 2)
        DEFAULT_MAX_EXTRA=16
        ;;
    deep10m)
        # `vamana/scripts/deep10M` (verbatim — PA's published recipe):
        #   BUILD_ARGS="-R 64 -L 128 -alpha 1.05 -num_passes 2 -quantize_bits 8 -verbose"
        #   QUERY_ARGS="-quantize_bits 16 -quantize_mode 1 -verbose -rerank_factor 2"
        #   TYPE_ARGS="-data_type float -dist_func Euclidian -file_type bin"
        BASE_FVECS="data/deep10m/deep10m_base.fvecs"
        QUERY_FVECS="data/deep10m/deep10m_query.fvecs"
        GT_IVECS="data/deep10m/deep10m_groundtruth.ivecs"
        PA_DIR_NAME="deep10M"
        DEFAULT_MAX_POINTS=""
        PA_BUILD_R=64
        PA_BUILD_L=128
        PA_BUILD_ALPHA=1.05
        PA_BUILD_PASSES=2
        PA_BUILD_QBITS=8
        PA_DIST_FUNC="Euclidian"
        PA_NORMALIZE_FLAG=""
        PA_QUERY_ARGS=(-quantize_bits 16 -quantize_mode 1 -verbose -rerank_factor 2)
        DEFAULT_MAX_EXTRA=16
        ;;
    fashion-mnist)
        # `vamana/scripts/fashion` (verbatim — PA's published recipe):
        #   BUILD_ARGS="-R 40 -L 80 -alpha 1.1 -num_passes 2 -quantize_bits 8 -verbose"
        #   QUERY_ARGS="-quantize_bits 8 -verbose"
        #   TYPE_ARGS="-data_type float -dist_func Euclidian -file_type bin"
        BASE_FVECS="data/fashion-mnist/fashion-mnist-784-euclidean_base.fvecs"
        QUERY_FVECS="data/fashion-mnist/fashion-mnist-784-euclidean_query.fvecs"
        GT_IVECS="data/fashion-mnist/fashion-mnist-784-euclidean_groundtruth.ivecs"
        PA_DIR_NAME="fashion-mnist-784-euclidean"
        DEFAULT_MAX_POINTS=""
        PA_BUILD_R=40
        PA_BUILD_L=80
        PA_BUILD_ALPHA=1.1
        PA_BUILD_PASSES=2
        PA_BUILD_QBITS=8
        PA_DIST_FUNC="Euclidian"
        PA_NORMALIZE_FLAG=""
        PA_QUERY_ARGS=(-quantize_bits 8 -verbose)
        DEFAULT_MAX_EXTRA=16
        ;;
    msmarco_bert_1M)
        # `vamana/scripts/msmarco_websearch` (PA's published MS-MARCO recipe):
        #   BUILD_ARGS="-R 64 -L 128 -alpha 1 -num_passes 1 -quantize_bits 8 -verbose"
        #   QUERY_ARGS="-quantize_bits 16 -quantize_mode 5 -verbose -rerank_factor 2"
        #   TYPE_ARGS="-data_type float -dist_func mips -file_type bin"
        BASE_FVECS="data/msmarco_bert_1M/msmarco_bert_1M_base.fvecs"
        QUERY_FVECS="data/msmarco_bert_1M/msmarco_bert_1M_query.fvecs"
        GT_IVECS="data/msmarco_bert_1M/msmarco_bert_1M_groundtruth.ivecs"
        PA_DIR_NAME="MSMarcoBert1M"
        DEFAULT_MAX_POINTS=""
        PA_BUILD_R=64
        PA_BUILD_L=128
        PA_BUILD_ALPHA=1.0
        PA_BUILD_PASSES=1
        PA_BUILD_QBITS=8
        PA_DIST_FUNC="mips"
        PA_NORMALIZE_FLAG=""        # GT is brute-force MIPS on raw f32
        PA_QUERY_ARGS=(-quantize_bits 16 -quantize_mode 5 -verbose -rerank_factor 2)
        DEFAULT_MAX_EXTRA=16
        ;;
    wiki_ada_1M)
        # OpenAI ada-002 + Wikipedia 1M (sourced from
        # nlpkevinl/wikipedia_openai_embeddings via load_wiki_ada_1M.py).
        # Build recipe: high-D shape (R=100 L=200 α=1.05 num_passes=2)
        # with `-dist_func mips` since ada-002 embeddings are
        # dot-product / cosine, not L2. ada-002 outputs are unit-norm
        # so MIPS == cosine ranking natively.
        BASE_FVECS="data/wiki_ada_1M/wiki_ada_1M_base.fvecs"
        QUERY_FVECS="data/wiki_ada_1M/wiki_ada_1M_query.fvecs"
        GT_IVECS="data/wiki_ada_1M/wiki_ada_1M_groundtruth.ivecs"
        PA_DIR_NAME="WikiAda1M"
        DEFAULT_MAX_POINTS=""
        PA_BUILD_R=100
        PA_BUILD_L=200
        PA_BUILD_ALPHA=1.05
        PA_BUILD_PASSES=2
        PA_BUILD_QBITS=""
        PA_DIST_FUNC="mips"
        PA_NORMALIZE_FLAG=""        # ada-002 outputs are already unit-norm
        PA_QUERY_ARGS=(-quantize_bits 16 -quantize_mode 5 -verbose -rerank_factor 2)
        DEFAULT_MAX_EXTRA=16
        ;;
    *)
        echo "Unknown DATASET=$DATASET (expected: sift | glove25 | glove100 | gist | deep10m | fashion-mnist | msmarco_bert_1M | wiki_ada_1M)" >&2
        exit 2
        ;;
esac
}

resolve_parlay_root() {
  PA_ROOT="${PA_ROOT:-../ParlayANN}"
  if [[ ! -d "$PA_ROOT" ]]; then
    echo "PA_ROOT is not a directory: $PA_ROOT; set it to your ParlayANN checkout" >&2
    return 2
  fi
  PA_ROOT="$(cd "$PA_ROOT" && pwd)"
  export PA_ROOT
}
