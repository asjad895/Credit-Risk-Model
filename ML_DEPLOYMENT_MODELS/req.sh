pip_requirements() {
if test "$#" -eq 0
then
  echo $'\nProvide at least one Python package name\n'
else
  for package in "$@"
  do
    pip install $package --user
    pip freeze | grep -i $package >> requirements.txt
  done
fi
}
pip_requirements pandas streamlit sklearn time base64 pickle urllib