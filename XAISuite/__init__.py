from pkg_resources import get_distribution, DistributionNotFound

try:
    dist = get_distribution("XAISuite")
except DistributionNotFound:
    __version__ = "Please install XAISuite with setup.py"
else:
    __version__ = dist.version
