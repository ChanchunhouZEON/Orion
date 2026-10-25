"""Run with DISKANN_BINARY pointing to the prebuilt executable."""
import tempfile,struct,subprocess,pathlib,os
binary=pathlib.Path(os.environ.get('DISKANN_BINARY', 'target/debug/diskann_sweep')).resolve()
with tempfile.TemporaryDirectory() as d:
 p=pathlib.Path(d)
 base=p/'base.bvecs'; query=p/'query.bvecs'; gt=p/'gt.ivecs'
 base.write_bytes(b''.join(struct.pack('<I',128)+bytes([i])*128 for i in range(128)))
 query.write_bytes(struct.pack('<I',128)+bytes([42])*128)
 gt.write_bytes(struct.pack('<II',1,42))
 command=[str(binary),'sift1b','--base',str(base),'--query',str(query),'--ground-truth',str(gt),'--cache-dir',str(p/'cache'),'--threads','2','--k','1','--search-list-sizes','64']
 for phase in ['build','cache']:
  result=subprocess.run(command,text=True,capture_output=True)
  assert result.returncode==0,result.stderr
  assert 'R@1=1.000000' in result.stdout,result.stdout
  if phase=='cache': assert 'loaded from cache' in result.stdout
  assert not list((p/'cache').glob('diskann-build-*'))
 # A malformed later base header would fail decoding. Restore mtime to keep
 # identity unchanged and prove a cache hit only inspects shape metadata.
 stat=base.stat(); data=bytearray(base.read_bytes()); data[132:136]=struct.pack('<I',127)
 base.write_bytes(data); os.utime(base,ns=(stat.st_atime_ns,stat.st_mtime_ns))
 result=subprocess.run(command,text=True,capture_output=True)
 assert result.returncode==0,result.stderr
 assert 'loaded from cache' in result.stdout
 print('PASS: build, search recall, cache reload without decoding base, temporary cleanup')
