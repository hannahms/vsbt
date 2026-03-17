#!/bin/bash
set -e

SUITE="${1:?Usage: $0 <config.yaml> [SB_LIST]}"
WORKDIR="/data/vsbt"

# Infer suite runner from config filename
case "$(basename "$SUITE")" in
  pgvector*)  RUNNER="pgvector_suite.py" ;;
  vectorchord*) RUNNER="vectorchord_suite.py" ;;
  pgpu*)      RUNNER="pgpu_suite.py" ;;
  *)          echo "Cannot infer suite from config: $SUITE"; exit 1 ;;
esac

# Default shared_buffers list, or pass as second arg (comma-separated)
if [ -n "$2" ]; then
  IFS=',' read -ra SB_LIST <<< "$2"
else
  SB_LIST=(700GB 512GB 256GB 128GB 64GB 32GB 16GB)
fi

echo "Config:  $SUITE"
echo "Runner:  $RUNNER"
echo "SB list: ${SB_LIST[*]}"
echo ""

for SB in "${SB_LIST[@]}"; do
  echo "============================================"
  echo "  Testing shared_buffers = $SB"
  echo "============================================"

  psql -U postgres -c "ALTER SYSTEM SET shared_buffers = '$SB'"
  psql -U postgres -c "ALTER SYSTEM SET maintenance_work_mem = '16GB'"
  systemctl restart postgresql-17
  sleep 10

  psql -U postgres -c "SHOW shared_buffers"

  echo "--- With page cache ---"
  cd $WORKDIR && python3 "$RUNNER" -s "$SUITE" --skip-add-embeddings --skip-index-creation

  echo "--- Without page cache ---"
  cd $WORKDIR && python3 "$RUNNER" -s "$SUITE" --skip-add-embeddings --skip-index-creation --no-fs-cache

  echo "Done with shared_buffers = $SB"
  echo ""
done

echo "All runs complete!"
