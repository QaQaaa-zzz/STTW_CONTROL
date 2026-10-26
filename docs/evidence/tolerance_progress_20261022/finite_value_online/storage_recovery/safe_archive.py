"""Verified byte-preserving transfer of stopped run payloads, then logical symlink."""
import hashlib,json,os,shutil,time
from pathlib import Path


def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for part in iter(lambda:stream.read(4*1024*1024),b''):h.update(part)
 return h.hexdigest()


def copy_verified_file(source,target):
 with source.open('rb') as inp,target.open('xb') as out:
  shutil.copyfileobj(inp,out,length=4*1024*1024);out.flush();os.fsync(out.fileno())


def save(path,state):
 temporary=path.with_name(path.name+'.tmp')
 with temporary.open('w') as stream:
  json.dump(state,stream,ensure_ascii=False,indent=2);stream.flush();os.fsync(stream.fileno())
 os.replace(temporary,path)
 fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)


def fsync_tree(target):
 directories=[p for p in target.rglob('*') if p.is_dir()]+[target,target.parent]
 for directory in sorted(directories,key=lambda p:len(p.parts),reverse=True):
  fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
  try:os.fsync(fd)
  finally:os.close(fd)


def archive(source,target,receipt):
 source=Path(source).absolute();target=Path(target).absolute();receipt=Path(receipt).absolute()
 if source.is_symlink() or not source.is_dir():raise ValueError('regular stopped source directory required')
 source_real=source.resolve();target_real=target.resolve();receipt_real=receipt.resolve()
 temporary_real=receipt.with_name(receipt.name+'.tmp').resolve()
 def nested(a,b):return a==b or a in b.parents
 if nested(source_real,target_real) or nested(target_real,source_real):raise ValueError('source and target must be disjoint')
 for file in [receipt_real,temporary_real]:
  if nested(source_real,file) or nested(target_real,file):raise ValueError('receipt must be outside both payload trees')
 if not target.parent.is_dir() or not receipt.parent.is_dir():raise ValueError('archive and receipt parents must already exist')

 entries=[]
 for path in sorted(source.rglob('*')):
  if path.is_symlink():raise ValueError('source symlinks must not be followed')
  if path.is_file():
   stat=path.stat();entries.append(dict(relative=str(path.relative_to(source)),size=stat.st_size,mtime_ns=stat.st_mtime_ns))
 if not entries:raise ValueError('nonempty source required')
 target.mkdir(parents=True,exist_ok=False);receipt.parent.mkdir(parents=True,exist_ok=True)
 state=dict(phase='copying',source=str(source),target=str(target),files=len(entries),bytes=sum(e['size'] for e in entries),copied_files=0,entries=entries,checked_unix=time.time())
 save(receipt,state)
 for entry in entries:
  src=source/entry['relative'];dst=target/entry['relative'];dst.parent.mkdir(parents=True,exist_ok=True)
  copy_verified_file(src,dst);sha=digest(src)
  if digest(dst)!=sha:raise ValueError('copy checksum differs: '+entry['relative'])
  entry['sha256']=sha;state['copied_files']+=1
  if state['copied_files']%10==0:state['checked_unix']=time.time();save(receipt,state)
 current=[str(p.relative_to(source)) for p in sorted(source.rglob('*')) if p.is_file()]
 if current!=[e['relative'] for e in entries]:raise ValueError('source file set changed during archive')
 for entry in entries:
  stat=(source/entry['relative']).stat()
  if (stat.st_size,stat.st_mtime_ns)!=(entry['size'],entry['mtime_ns']):raise ValueError('source changed during archive')
 fsync_tree(target)
 state.update(phase='verified_copy_before_source_removal',checked_unix=time.time());save(receipt,state)
 # Every payload now exists at the fsynced destination with matching SHA.
 # Remove only this verified stopped payload; no other directory is touched.
 try:
  shutil.rmtree(source);source.symlink_to(target,target_is_directory=True)
  parent_fd=os.open(source.parent,os.O_RDONLY|os.O_DIRECTORY)
  try:os.fsync(parent_fd)
  finally:os.close(parent_fd)
 except BaseException:
  state.update(phase='verified_copy_original_path_repair_needed',checked_unix=time.time());save(receipt,state);raise
 state.update(phase='archived_original_path_preserved',checked_unix=time.time());save(receipt,state)
 return state
