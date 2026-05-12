rm -rf _skbuild
rm -rf fbgemm_ascend.egg-info/
pip uninstall fbgemm_ascend -y

pip install . --no-build-isolation