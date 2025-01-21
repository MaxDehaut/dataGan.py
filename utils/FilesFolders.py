import pathlib

def get_files(pathtodir, extension="*.*"):
    """Retrieves the list of files with a directory"""
    if not isinstance(pathtodir, pathlib.PurePath):
        pathtodir = pathlib.Path(pathtodir)
    return sorted(list(pathtodir.glob(extension)))


def get_file_list(pathtodir, extension="*.*"):
    """Returns list of files from a folder"""
    raw_files = get_files(pathtodir, extension)

    files = []
    for file_path in raw_files:
        files.append(file_path.name)

    return files


# -----------
# - FOLDERS -
# -----------


def get_folders(pathtodir, prefix=""):
    """Retrieves the list of folders with a directory."""
    if not isinstance(pathtodir, pathlib.PurePath):
        pathtodir = pathlib.Path(pathtodir)
    return sorted(
        [fld for fld in pathtodir.iterdir() 
         if fld.is_dir() and not fld.name.lower().startswith(prefix)])


def get_folder_names(pathtodir, prefix=""):
    """Retrieves the list of folders with a directory"""
    if not isinstance(pathtodir, pathlib.PurePath):
        pathtodir = pathlib.Path(pathtodir)
    return sorted(
        [fld.name for fld in pathtodir.iterdir() 
         if fld.is_dir() and not fld.name.lower().startswith(prefix)])
