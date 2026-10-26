import importlib.util,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('safe_archive',str(Path(__file__).with_name('safe_archive.py')))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class ArchiveTests(unittest.TestCase):
 def test_verified_bytes_survive_at_original_logical_path(self):
  with tempfile.TemporaryDirectory(dir='/dev/shm') as folder:
   root=Path(folder);src=root/'source';src.mkdir();(src/'x').write_bytes(b'alpha\x00beta');(src/'nested').mkdir();(src/'nested/y').write_bytes(b'gamma')
   result=module.archive(src,root/'target',root/'receipt.json')
   self.assertTrue(src.is_symlink());self.assertEqual((src/'x').read_bytes(),b'alpha\x00beta');self.assertEqual((src/'nested/y').read_bytes(),b'gamma');self.assertEqual(result['files'],2)
 def test_corrupt_copy_never_removes_source(self):
  with tempfile.TemporaryDirectory(dir='/dev/shm') as folder:
   root=Path(folder);src=root/'source';src.mkdir();(src/'x').write_bytes(b'original')
   with patch.object(module,'copy_verified_file',side_effect=lambda a,b:b.write_bytes(b'corrupt')):
    with self.assertRaises(ValueError):module.archive(src,root/'target',root/'receipt.json')
   self.assertFalse(src.is_symlink());self.assertEqual((src/'x').read_bytes(),b'original')
 def test_symlinks_and_existing_destination_are_not_followed_or_overwritten(self):
  with tempfile.TemporaryDirectory(dir='/dev/shm') as folder:
   root=Path(folder);src=root/'source';src.mkdir();outside=root/'outside';outside.write_bytes(b'keep');(src/'link').symlink_to(outside)
   with self.assertRaises(ValueError):module.archive(src,root/'target',root/'receipt.json')
   self.assertEqual(outside.read_bytes(),b'keep')
  with tempfile.TemporaryDirectory(dir='/dev/shm') as folder:
   root=Path(folder);src=root/'source';src.mkdir();(src/'x').write_bytes(b'x');target=root/'target';target.mkdir();(target/'keep').write_bytes(b'keep')
   with self.assertRaises(FileExistsError):module.archive(src,target,root/'receipt.json')
   self.assertEqual((target/'keep').read_bytes(),b'keep');self.assertTrue(src.is_dir())

 def test_directory_flush_failure_keeps_original_source(self):
  with tempfile.TemporaryDirectory(dir='/dev/shm') as folder:
   root=Path(folder);src=root/'source';src.mkdir();(src/'x').write_bytes(b'original')
   with patch.object(module,'fsync_tree',side_effect=OSError('directory fsync unavailable')):
    with self.assertRaises(OSError):module.archive(src,root/'target',root/'receipt.json')
   self.assertFalse(src.is_symlink());self.assertEqual((src/'x').read_bytes(),b'original')


 def test_overlapping_paths_never_overwrite_source(self):
  with tempfile.TemporaryDirectory(dir='/dev/shm') as folder:
   root=Path(folder);src=root/'source';src.mkdir();payload=src/'payload';payload.write_bytes(b'keep')
   for target,receipt in [(src/'nested',root/'receipt'),(root/'target',payload),(root/'target',src/'note')]:
    with self.assertRaises(ValueError):module.archive(src,target,receipt)
    self.assertEqual(payload.read_bytes(),b'keep')
 def test_missing_archive_parent_leaves_source_intact(self):
  with tempfile.TemporaryDirectory(dir='/dev/shm') as folder:
   root=Path(folder);src=root/'source';src.mkdir();(src/'x').write_bytes(b'keep')
   with self.assertRaises(ValueError):module.archive(src,root/'missing/target',root/'receipt')
   self.assertEqual((src/'x').read_bytes(),b'keep')

if __name__=='__main__':unittest.main()
