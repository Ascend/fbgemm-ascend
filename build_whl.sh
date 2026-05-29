rm -rf _skbuild/
rm -rf fbgemm_ascend.egg-info/
rm -rf dist/

pip wheel . --no-build-isolation --no-deps -v -w dist/ 2>&1
