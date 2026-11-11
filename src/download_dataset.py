
import urllib.request

url = "https://cs.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
destination = "data/stl10_binary.tar.gz"

print("Starting STL-10 download...")

urllib.request.urlretrieve(url, destination)

print("Download completed!")
