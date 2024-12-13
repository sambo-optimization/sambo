#!/bin/bash
set -eu

die () { echo "ERROR: $*" >&2; exit 2; }

for cmd in pdoc3; do
    command -v "$cmd" >/dev/null ||
        die "Missing $cmd; \`pip install $cmd\`"
done

out="www/doc"
package="sambo"


echo
echo 'Building API reference docs'
echo
pdoc3 --html --force \
     --template-dir ".github/pdoc_template" \
     --output-dir "$out" \
     "$package"


ANALYTICS1="<script async src='https://www.googletagmanager.com/gtag/js?id=G-QJH7PLMB12'></script><script>window.dataLayer=window.dataLayer||[];function gtag(){dataLayer.push(arguments);}gtag('js',new Date());gtag('config','G-QJH7PLMB12');</script>"
ANALYTICS2='<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-2900001379782823" crossorigin></script>'
find "$out" -name '*.html' -print0 |
    xargs -0 -- sed -i "s#</head>#$ANALYTICS1$ANALYTICS2</head>#i"

exit 0  # TODO: For now

echo
echo 'Testing for broken links'
echo
problematic_urls='
https://www.gnu.org/licenses/agpl-3.0.html
'
pushd "$out" >/dev/null
grep -PR '<a .*?href=' |
    sed -E "s/:.*?<a .*?href=([\"'])(.*?)/\t\2/g" |
    tr "\"'" '#' |
    cut -d'#' -f1 |
    sort -u -t$'\t' -k 2 |
    sort -u |
    python -c '
import sys
from urllib.parse import urljoin
for line in sys.stdin.readlines():
    base, url = line.split("\t")
    print(base, urljoin(base, url.strip()), sep="\t")
    ' |
    grep -v $'\t''$' |
    while read -r line; do
        while IFS=$'\t' read -r file url; do
            echo "$file: $url"
            [ -f "$url" ] ||
                curl --silent --fail --retry 3 --retry-delay 1 --connect-timeout 10 \
                        --user-agent 'Mozilla/5.0 Firefox 125' "$url" >/dev/null 2>&1 ||
                    grep -qF "$url" <(echo "$problematic_urls") ||
                    die "broken link in $file:  $url"
        done
    done
popd >/dev/null


echo
echo "All good. Docs in: $out"
echo
echo "    file://$(readlink -f "$out")/$package/index.html"
echo
