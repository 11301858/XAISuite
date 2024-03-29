#!/bin/bash
set -euo pipefail

# Change to root directory of repo
DIRNAME=$(cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
cd "${DIRNAME}/.."

# Set up virtual environment
pip3 install setuptools wheel virtualenv
if [ ! -d venv ]; then
  rm -f venv
  virtualenv venv
fi
source venv/bin/activate

# Get current git branch & stash unsaved changes
GIT_BRANCH=$(git branch --show-current)
if [ -z "${GIT_BRANCH}" ]; then
    GIT_BRANCH="main"
fi
git stash

# Set up exit handler to restore git state & delete temp branches
function exit_handler {
    git reset --hard
    git checkout "${GIT_BRANCH}" --
    git stash pop || true
    for version in $(git tag --list 'v[0-9]*'); do
        branch="${version}_local_docs_only"
        if git show-ref --verify --quiet "refs/heads/$branch"; then
            git branch -D "$branch"
        fi
    done
}
trap exit_handler EXIT

# Clean up build directory and install Sphinx requirements
pip3 install -r "${DIRNAME}/requirements.txt"
sphinx-build -M clean "${DIRNAME}" "${DIRNAME}/_build"

# Build API docs for current head
export current_version="latest"
pip3 install ".[all]"
sphinx-build -b html "${DIRNAME}" "${DIRNAME}/_build/html/${current_version}" -W --keep-going
rm -rf "${DIRNAME}/_build/html/${current_version}/.doctrees"
pip3 uninstall -y xaisuite

# Install all previous released versions
# and use them to build the appropriate API docs.
# Uninstall after we're done with each one.
versions=()
checkout_files=("${DIRNAME}/*.rst" "demo" "xaisuite" "xaisuitegui" "setup.py")
for version in $(git tag --list 'v[0-9]*'); do
    if [[ "$version" != "v2.9.2" ]]; then #Skip intermediate versions
      echo Skipping $version
      continue
    fi
    versions+=("$version")
    git fetch
    git checkout -b "${version}_local_docs_only"
    for f in $(git diff --name-only --diff-filter=A "tags/${version}" "${DIRNAME}/*.rst"); do
        git fetch
        git rm "$f"
    done
    git fetch
    git checkout "tags/${version}" -- "${checkout_files[@]}"
    export current_version=${version} 
    pip3 install ".[all]"
    sphinx-build -b html "${DIRNAME}" "${DIRNAME}/_build/html/${current_version}" -W --keep-going
    rm -rf "${DIRNAME}/_build/html/${current_version}/.doctrees"
    pip3 uninstall -y xaisuite
    git reset --hard
    git fetch
    git checkout "${GIT_BRANCH}" --
done

# Determine the latest stable version if there is one
if (( ${#versions[@]} > 0 )); then
  stable_hash=$(git rev-list --tags --max-count=1)
  stable_version=$(git describe --tags "$stable_hash")
  export stable_version
else
  export stable_version="latest"
fi

# Create dummy HTML's for the stable version in the base directory
while read -r filename; do
    filename=$(echo "$filename" | sed "s/\.\///")
    n_sub=$(echo "$filename" | (grep -o "/" || true) | wc -l)
    prefix=""
    for (( i=0; i<n_sub; i++ )); do
        prefix+="../"
    done
    url="${prefix}${stable_version}/$filename"
    mkdir -p "${DIRNAME}/_build/html/$(dirname "$filename")"
    cat > "${DIRNAME}/_build/html/$filename" <<EOF
<!DOCTYPE html>
<html>
   <head>
      <title>XAISuite's Documentation</title>
      <meta http-equiv = "refresh" content="0; url='$url'" />
   </head>
   <body>
      <p>Please wait while you're redirected to our <a href="$url">documentation</a>.</p>
   </body>
</html>
EOF
done < <(cd "${DIRNAME}/_build/html/$stable_version" && find . -name "*.html")
echo "Finished writing to _build/html." 
